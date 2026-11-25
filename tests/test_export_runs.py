import unittest

from scripts.export_monitoring import EXPORTS
from scripts.export_runs import RETRAINING_OUTPUT_PATH, _portable_artifact_uri


class ExportRunsTests(unittest.TestCase):
    def test_artifact_uri_is_relative_to_the_project(self) -> None:
        self.assertEqual(
            _portable_artifact_uri(4, "run-123"),
            "mlartifacts/4/run-123/artifacts",
        )

    def test_operational_and_mlflow_retraining_exports_do_not_collide(self) -> None:
        self.assertNotIn(RETRAINING_OUTPUT_PATH.name, EXPORTS)
        self.assertIn("reentrenamientos_operativos.csv", EXPORTS)


if __name__ == "__main__":
    unittest.main()
