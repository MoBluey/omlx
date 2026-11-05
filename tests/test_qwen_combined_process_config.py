"""CPU-only checks of import-time opt-in assignments, without MLX imports."""

import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch


class CombinedProcessConfigTests(unittest.TestCase):
    def test_explicit_opt_in_initialises_all_three_controls(self):
        repo_path = Path(__file__).resolve().parents[1]
        flag = "OMLX_EXPERIMENTAL_QWEN_COMBINED_MARGINAL"
        controls = {
            "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/qsa_fast.py": {
                "_DECODE_SELECT_REDUCED_BARRIERS_ENABLED": (False, True),
                "_FUSED_DECODE_KV_GATHER": (False, True),
            },
            "omlx/patches/mlx_lm_mtp/context_copy.py": {
                "_MAX_MATCH": (64, 32),
            },
        }
        for relative_path, expected in controls.items():
            source_path = repo_path / relative_path
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            assignments = [
                node for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id in expected
                        for target in node.targets)
            ]
            self.assertEqual(len(assignments), len(expected))
            code = compile(ast.Module(body=assignments, type_ignores=[]), str(source_path), "exec")
            for value in (None, "", "0", "true", "1", " 1"):
                with self.subTest(module=relative_path, value=value), patch.dict(os.environ):
                    if value is None:
                        os.environ.pop(flag, None)
                    else:
                        os.environ[flag] = value
                    namespace = {"os": os}
                    exec(code, namespace)
                    for name, values in expected.items():
                        self.assertEqual(namespace[name], values[value == "1"])


if __name__ == "__main__":
    unittest.main()
