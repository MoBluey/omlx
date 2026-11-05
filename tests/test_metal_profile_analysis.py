"""CPU-only checks for attribution arithmetic, not implementation mirrors."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("analyse", Path(__file__).parents[1] / "scripts/analyse_metal_profile.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AttributionTests(unittest.TestCase):
    def test_overlap_mixed_and_invalid(self):
        report = module.analyse([
            {"gpu_start": 1, "gpu_end": 3, "status": 4, "kernels": ["a", "b"]},
            {"gpu_start": 2, "gpu_end": 4, "status": 4, "kernels": ["c"]},
            {"gpu_start": 0, "gpu_end": 1, "status": 4, "kernels": ["a"]},
            {"gpu_start": 5, "gpu_end": 5, "status": 4, "kernels": []},
            {"gpu_start": 5, "gpu_end": 7, "status": 5, "kernels": ["a"]},
        ])
        self.assertEqual(report["valid_buffers"], 2)
        self.assertEqual(report["invalid_or_zero_buffers"], 3)
        self.assertEqual(report["summed_buffer_seconds"], 4)
        self.assertEqual(report["gpu_interval_union_seconds"], 3)
        self.assertEqual(report["overlap_seconds"], 1)
        self.assertEqual(len(report["groups"]), 2)
        self.assertEqual(report["groups"][0]["kernels"], ("a", "b"))
        self.assertEqual(report["groups"][0]["share_of_summed_buffer_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
