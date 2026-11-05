"""Opt-in diagnostic only: exact live-tail -> restored-prefix state checks.

The hashes force array evaluation and copy bytes to the CPU. Never enable in
performance runs. Hooks observe normal production cache producers/consumers;
they do not seed cache entries, change prompt tokens or forward the model.
"""

from functools import wraps
import hashlib
import json
import os
from pathlib import Path


def _token_digest(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def _different(a, b, path=''):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [path + '.keys']
        return [p for key in a for p in _different(a[key], b[key], path + '.' + str(key))]
    if isinstance(a, list):
        if len(a) != len(b):
            return [path + '.length']
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _different(x, y, path + '[' + str(i) + ']')]
    return [] if a == b else [path]


def install():
    if os.environ.get('QWEN_TRIAL_IDLE_STATE') != '1':
        return
    import mlx.core as mx
    from omlx.scheduler import Scheduler
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.patches.mlx_lm_mtp import prompt_priming

    output = Path(os.environ['QWEN_PROBE_DIR']) / 'idle-state.json'
    captures = {}
    head_captures = {}
    events = []

    def record(event):
        events.append(event)
        output.write_text(json.dumps({'diagnostic_only': True, 'events': events}, indent=2))

    def value(v):
        if isinstance(v, mx.array):
            # uint8 view retains the exact representation, including bfloat16.
            raw = v.reshape(-1).view(mx.uint8)
            mx.eval(raw)
            return {'shape': list(v.shape), 'dtype': str(v.dtype),
                    'sha256': hashlib.sha256(memoryview(raw).tobytes()).hexdigest()}
        if isinstance(v, (tuple, list)):
            return [value(x) for x in v]
        if isinstance(v, dict):
            return {str(k): value(x) for k, x in v.items()}
        if v is None or isinstance(v, (str, bool, int, float)):
            return v
        raise TypeError('unsupported state leaf ' + type(v).__name__)

    def cache_state(cache):
        result = []
        for entry in cache:
            item = {'class': type(entry).__name__, 'state': value(entry.state),
                    'meta': value(getattr(entry, 'meta_state', ())),
                    'offset': value(getattr(entry, 'offset', None))}
            if hasattr(entry, 'caches'):
                item['members'] = cache_state(entry.caches)
            result.append(item)
        return result

    def final_prefill(self, request, cache, last_tokens, route):
        with mx.stream(self._stream):
            state = cache_state(cache)
            ctx = prompt_priming._find_ctx(self.model)
            head = None if ctx is None else {
                'folded': ctx.folded, 'expected_offset': ctx.expected_offset,
                'valid': ctx.valid, 'cache': cache_state(ctx.mtp_cache),
                'pending_hidden': value(ctx.pending_hidden),
            }
        record({'kind': 'final_prefill', 'request_id': request.request_id,
                'prompt_sha256': _token_digest(request.prompt_token_ids),
                'prompt_tokens': len(request.prompt_token_ids),
                'prefilled_tokens': len(request.prompt_token_ids)-len(last_tokens),
                'last_tokens': list(last_tokens), 'cached_tokens': request.cached_tokens,
                'route': route, 'state': state, 'head': head})

    external_prefill = Scheduler._do_external_prefill

    @wraps(external_prefill)
    def external(self, request, tokens, existing_cache, *args, **kwargs):
        cache, last_tokens = external_prefill(self, request, tokens, existing_cache, *args, **kwargs)
        final_prefill(self, request, cache, last_tokens, 'external')
        return cache, last_tokens

    Scheduler._do_external_prefill = external
    insert_prefilled = Scheduler._insert_prefilled_request

    @wraps(insert_prefilled)
    def insert(self, request, state, scheduled):
        final_prefill(self, request, state.cache, state.last_token, 'chunked')
        return insert_prefilled(self, request, state, scheduled)

    Scheduler._insert_prefilled_request = insert
    build_sampler = Scheduler._build_sampler_and_processors

    @wraps(build_sampler)
    def sampler_for(self, sampling_params, request=None):
        sampler, processors = build_sampler(self, sampling_params, request)
        if request is None:
            return sampler, processors
        captured_first = False

        @wraps(sampler)
        def sampled(logprobs):
            nonlocal captured_first
            if captured_first:
                return sampler(logprobs)
            captured_first = True
            # Preserve sampler metadata through functools.wraps; introducing
            # a logits processor here could silently disable the MTP route.
            rng_before = value(list(mx.random.state))
            result = sampler(logprobs)
            mx.eval(result)
            rng_after = value(list(mx.random.state))
            prompt_hash = _token_digest(request.prompt_token_ids)
            request_hash = hashlib.sha256(request.request_id.encode()).hexdigest()[:12]
            name = 'idle-first-sample-' + prompt_hash + '-' + request_hash + '.npy'
            mx.save(output.parent / name, logprobs.astype(mx.float32))
            record({'kind': 'first_sample', 'request_id': request.request_id,
                    'prompt_sha256': prompt_hash, 'seed': sampling_params.seed,
                    'sampling_logprobs': value(logprobs), 'logprobs_file': name,
                    'sampled_token_ids': result.tolist(),
                    'rng_before': rng_before, 'rng_after': rng_after})
            return result

        return sampled, processors

    Scheduler._build_sampler_and_processors = sampler_for

    def capture_at(self, request, prompt_cache, total_tokens, kind):
        # Capture every layer, including sliceable KV that normal boundary
        # snapshots replace with placeholders. PLE ArraysCache slots2/3 and
        # QSA index keys/position IDs are included through their .state.
        with mx.stream(self._stream):
            state = cache_state(prompt_cache)
        key = (total_tokens, _token_digest(request.prompt_token_ids[:total_tokens]))
        captures[key] = (request.request_id, state)
        record({'kind': kind, 'request_id': request.request_id,
                'tokens': total_tokens, 'prefix_sha256': key[1], 'state': state})

    boundary = Scheduler._emit_prefill_boundary_snapshot

    @wraps(boundary)
    def capture_boundary(self, request, prompt_cache, total_tokens):
        capture_at(self, request, prompt_cache, total_tokens, 'boundary_capture')
        return boundary(self, request, prompt_cache, total_tokens)

    Scheduler._emit_prefill_boundary_snapshot = capture_boundary
    tail = Scheduler._emit_prefill_tail_snapshot

    @wraps(tail)
    def capture(self, request, prompt_cache, total_tokens):
        if total_tokens % self.config.paged_cache_block_size:
            capture_at(self, request, prompt_cache, total_tokens, 'tail_capture')
        return tail(self, request, prompt_cache, total_tokens)

    Scheduler._emit_prefill_tail_snapshot = capture
    prepare = Scheduler._prepare_prefix_cache_for_request

    @wraps(prepare)
    def restored(self, request):
        result = prepare(self, request)
        n = int(request.cached_tokens or 0)
        if n > 0 and request.prompt_cache:
            key = (n, _token_digest(request.prompt_token_ids[:n]))
            prior = captures.get(key)
            if prior is not None and prior[0] != request.request_id:
                with mx.stream(self._stream):
                    actual = cache_state(request.prompt_cache)
                differences = _different(prior[1], actual)
                record({'kind': 'trunk_restore', 'request_id': request.request_id,
                        'capture_request_id': prior[0], 'tokens': n,
                        'prefix_sha256': key[1], 'exact': not differences,
                        'differences': differences, 'state': actual})
        return result

    Scheduler._prepare_prefix_cache_for_request = restored
    store_head = BlockAwarePrefixCache.store_mtp_prefix_snapshot

    @wraps(store_head)
    def stored_head(self, tokens, boundary_tokens, snapshot, **kwargs):
        result = store_head(self, tokens, boundary_tokens, snapshot, **kwargs)
        if result:
            key = (boundary_tokens, _token_digest(tokens[:boundary_tokens]))
            state = {'cache': cache_state(snapshot.mtp_cache),
                     'pending_hidden': value(snapshot.pending_hidden)}
            head_captures[key] = state
            record({'kind': 'head_capture', 'tokens': boundary_tokens,
                    'prefix_sha256': key[1], 'state': state})
        return result

    BlockAwarePrefixCache.store_mtp_prefix_snapshot = stored_head
    prepare_head = prompt_priming.prepare_prefix_context

    @wraps(prepare_head)
    def restored_head(model, *, request_id, **kwargs):
        result = prepare_head(model, request_id=request_id, **kwargs)
        n = int(kwargs['cached_tokens'] or 0)
        key = (n, _token_digest(kwargs['prompt_tokens'][:n]))
        prior = head_captures.get(key)
        if n > 0:
            ctx = prompt_priming._find_ctx(model)
            event = {'kind': 'head_restore', 'request_id': request_id,
                     'tokens': n, 'prefix_sha256': key[1],
                     'sidecar_found': prior is not None, 'restored': bool(result)}
            if result and ctx is not None and prior is not None:
                actual = {'cache': cache_state(ctx.mtp_cache),
                          'pending_hidden': value(ctx.pending_hidden)}
                differences = _different(prior, actual)
                event.update({'exact': not differences, 'differences': differences,
                              'folded': ctx.folded, 'expected_offset': ctx.expected_offset,
                              'positions_exact': ctx.folded == n-1 and ctx.expected_offset == n,
                              'state': actual})
            record(event)
        return result

    prompt_priming.prepare_prefix_context = restored_head
    record({'kind': 'installed'})
    return {'value': value, 'cache_state': cache_state}
