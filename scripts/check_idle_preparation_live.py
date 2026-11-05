"""Bounded actual-HTTP qualification inside main's guarded private server.

Arguments: absolute private run directory, owned server PID. The server must
enable OMLX_EXPERIMENTAL_QWEN_IDLE_BOUNDARY and idle_live_observer.install().
Reuses the existing streaming probe; starts no server/executor/model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
WORKSPACE = Path('/Users/mohameddarwiche/work/p2pit/development/QwenLocal')
BENCH = WORKSPACE / 'state/benchmarks/qwen-idle-expert-20261105'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('pid', type=int)
    args = parser.parse_args()
    out = args.directory
    assert out.is_absolute() and out.is_dir()
    sys.path[:0] = [str(out/'harness'), str(WORKSPACE/'scripts')]
    import ornith_perf_ab as probe
    import ttft_probe
    os.kill(args.pid, 0)
    key = (out/'api-key').read_text().strip()
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    ttft_probe.install(probe)
    probe.PORT = 8011
    probe.MODEL_ID = 'Qwen3.8-Flash-Next-Uncensored-oQ4e-100K-MTP'
    controls = {'temperature':1.0, 'top_p':0.95, 'top_k':20, 'seed':20261106,
                'reasoning_effort':'medium', 'chat_template_kwargs':{'enable_thinking':True},
                'max_tokens':8192}
    records = []

    def save():
        (out/'idle-live-results.json').write_text(json.dumps({'requests':records}, indent=2))

    def events():
        result = []
        for line in (out/'server.log').read_text().splitlines():
            offset = line.find('{')
            if offset >= 0:
                try:
                    data = json.loads(line[offset:])
                    if data.get('event') == 'idle_prefix_preparation': result.append(data)
                except ValueError:
                    pass
        return result

    def owner():
        path = out/'idle-live-owner.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def wait(read, predicate, timeout=30):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            os.kill(args.pid, 0)
            for item in reversed(read()):
                if predicate(item): return item
            time.sleep(0.01)
        raise AssertionError('idle live observation timed out')

    def chat(label, messages, *, stream=True, overrides=None):
        payload = {'model':probe.MODEL_ID, 'stream':stream, 'messages':messages,
                   **controls, **(overrides or {})}
        if stream:
            payload['stream_options'] = {'include_usage':True}
        (out/('idle-live-'+label+'-request.json')).write_text(json.dumps(payload, indent=2))
        started = time.time()
        print(json.dumps({'event':'idle_live_request','label':label,'at':started}), flush=True)
        if stream:
            result = probe.chat(key, payload, timeout=600)
        else:
            wire = {'model':probe.MODEL_ID,'stream':False,**payload}
            request = urllib.request.Request('http://127.0.0.1:8011/v1/chat/completions',
                data=json.dumps(wire).encode(), headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=600) as response:
                body = json.load(response)
            (out/('idle-live-'+label+'-response.json')).write_text(json.dumps(body, indent=2))
            choice = body['choices'][0]
            usage = body['usage']
            text = choice['message'].get('content') or ''
            reasoning = choice['message'].get('reasoning_content') or ''
            result = {'text':text,'finish_reason':choice['finish_reason'],
                      'reasoning_sha256':hashlib.sha256(reasoning.encode()).hexdigest(),
                      'prompt_tokens':usage['prompt_tokens'],'completion_tokens':usage['completion_tokens'],
                      'cached_tokens':usage.get('prompt_tokens_details',{}).get('cached_tokens',0)}
        result['answer_sha256'] = hashlib.sha256(result['text'].encode()).hexdigest()
        records.append({'label':label,'started':started,'ended':time.time(),'result':result})
        save()
        return result

    def initial(session):
        return [{'role':'system','content':f'Profiling session 115000-{session:032x}. Reference code:\n\n'+probe.corpus(115000)},
                {'role':'user','content':'Explain the model startup and cleanup sequence in the reference code. Give four concise numbered points using the relevant function names.'}]

    def interrupt(label):
        # A deliberately incomplete foreground turn cannot schedule new work.
        return chat(label,[{'role':'user','content':'Consider why careful cleanup matters.'}],
                    overrides={'max_tokens':1})

    control = out/'idle-live-control.json'
    control.write_text(json.dumps({'queued_gate':False}))
    start_events = len(events())
    history = initial(2)
    cold = chat('success-cold',history)
    assert cold['finish_reason'] == 'stop' and cold['text'].strip()
    completed = wait(lambda:events()[start_events:],lambda x:x['status']=='completed')
    assert completed['chosen_boundary'] == 28672
    history += [{'role':'assistant','content':cold['text']},
                {'role':'user','content':'Which cleanup step matters most after a failed launch? Answer in one sentence.'}]
    followup = chat('success-followup',history)
    baseline = json.loads((BENCH/'boundary-baseline-timing/qwen-ttft-long-1-reused.json').read_text())
    assert followup['cached_tokens'] == 28672
    assert followup['answer_sha256'] == baseline['output_sha256']
    assert followup['reasoning_sha256'] == baseline['result']['reasoning_sha256']
    interrupt('after-success-drain')

    # The observation-only queued gate holds the async background task, not
    # the owner thread or any GPU work. It never seeds cache/state/output.
    control.write_text(json.dumps({'queued_gate':True}))
    old_owner = len(owner())
    start_events = len(events())
    queued = chat('queued-nonstream-cold',initial(4),stream=False)
    assert queued['finish_reason']=='stop'
    wait(lambda:owner()[old_owner:],lambda x:x['kind']=='queued_gate_enter')
    interrupt('queued-foreground')
    wait(lambda:events()[start_events:],lambda x:x['status']=='cancelled')
    assert not any(x['kind']=='idle_prefill_enter' for x in owner()[old_owner:])
    control.write_text(json.dumps({'queued_gate':False}))

    old_owner = len(owner())
    active_history = initial(5)
    active_history[-1]['content'] = ('Explain the model startup and cleanup sequence in the reference code. '
        'Give eight detailed numbered points, retaining relevant function names and explaining failure handling.')
    active = chat('active-cold',active_history)
    assert active['finish_reason']=='stop'
    entered = wait(lambda:owner()[old_owner:],lambda x:x['kind']=='idle_prefill_enter')
    interrupt('active-foreground')
    cancelled = wait(lambda:owner()[old_owner:],lambda x:x['kind']=='idle_cancel_requested' and x['request_id']==entered['request_id'])
    exited = wait(lambda:owner()[old_owner:],lambda x:x['kind']=='idle_prefill_exit' and x['request_id']==entered['request_id'])
    cleaned = wait(lambda:owner()[old_owner:],lambda x:x['kind']=='idle_abort_cleanup' and x['request_id']==entered['request_id'])
    assert entered['at'] <= cancelled['at'] < exited['at'], (entered,cancelled,exited)
    assert exited['error']=='_PrefillAbortedError'
    assert not cleaned['request_retained'] and not cleaned['transient_snapshot_retained']
    before_drain = len(owner())
    interrupt('final-drain')
    released = wait(lambda:owner()[before_drain:],lambda x:x['kind']=='foreground_released')
    assert released['active_foreground']==0 and not released['pending_idle'] and not released['idle_task_active']
    summary = {'success':True,'successful_prefix':28672,'reasoning_and_answer_exact':True,
               'nonstream_scheduled':True,'queued_cancelled_before_owner_prefill':True,
               'active_cancelled_during_owner_prefill':True,'cancelled_transient_state_released':True,
               'no_background_task_active':True,
               'requests':len(records),'pid':args.pid,'tools_declaring_requests_eligible':False}
    (out/'idle-live-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'event':'idle_live_complete',**summary}),flush=True)


if __name__ == '__main__':
    main()
