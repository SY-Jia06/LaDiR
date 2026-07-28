from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.run_experiment_matrix import load_jobs


class ExperimentMatrixTest(unittest.TestCase):
    def test_cartesian_expansion_and_derived_arguments(self):
        yaml = """
groups:
  demo:
    command: python train.py
    name_template: run-{a}-{b}
    fixed_args: [fixed=1]
    dimensions:
      a:
        values: [x, y]
        arguments: ["model.a={value}"]
      b:
        values: [1, 2]
        arguments: ["model.b={value}"]
    derived_args:
      - trainer.output_dir=out/{name}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.yaml"
            path.write_text(yaml)
            jobs = load_jobs(str(path), {"demo"})
        self.assertEqual(len(jobs), 4)
        self.assertEqual(jobs[0].name, "run-x-1")
        self.assertIn("model.a=x", jobs[0].command)
        self.assertIn("model.b=1", jobs[0].command)
        self.assertIn("trainer.output_dir=out/run-x-1", jobs[0].command)


if __name__ == "__main__":
    unittest.main()
