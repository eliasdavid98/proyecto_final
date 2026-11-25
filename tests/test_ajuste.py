import unittest

from sklearn.model_selection import ParameterGrid

from src.ajuste import CONFIGURATIONS_PER_MODEL, tuning_specs


class HyperparameterTuningGridTests(unittest.TestCase):
    def test_top_three_models_each_have_exactly_30_configurations(self) -> None:
        specs = tuning_specs()

        self.assertEqual(
            [name for name, _, _ in specs],
            ["gradient_boosting", "random_forest", "svc_rbf"],
        )
        for model_name, _, param_grid in specs:
            with self.subTest(model=model_name):
                self.assertEqual(
                    len(list(ParameterGrid(param_grid))),
                    CONFIGURATIONS_PER_MODEL,
                )
                self.assertEqual(CONFIGURATIONS_PER_MODEL, 30)


if __name__ == "__main__":
    unittest.main()
