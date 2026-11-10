
import unittest
import matplotlib
import numpy as np

# Use a non-interactive backend for automated testing
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.utils import evaluate_model, plot_predictions


class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([100, 105, 110, 115, 120])
        self.y_pred = np.array([102, 103, 112, 113, 125])

    def test_mae(self):
        metrics = evaluate_model(self.y_true, self.y_pred)
        self.assertAlmostEqual(metrics["mae"], 2.6, places=4)

    def test_mse(self):
        metrics = evaluate_model(self.y_true, self.y_pred)
        self.assertAlmostEqual(metrics["mse"], 8.2, places=4)

    def test_rmse(self):
        metrics = evaluate_model(self.y_true, self.y_pred)
        self.assertAlmostEqual(metrics["rmse"], 8.2 ** 0.5, places=4)

    def test_r2_score(self):
        metrics = evaluate_model(self.y_true, self.y_pred)
        self.assertAlmostEqual(metrics["r2"], 0.836, places=4)

    def test_all_metrics_are_returned(self):
        metrics = evaluate_model(self.y_true, self.y_pred)

        expected_keys = {"mae", "mse", "rmse", "r2"}

        self.assertEqual(set(metrics.keys()), expected_keys)


class TestPlotPredictions(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([100, 105, 110, 115, 120])
        self.y_pred = np.array([102, 103, 112, 113, 125])

        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_plot_predictions(self):
        plot_predictions(self.y_true, self.y_pred)

        # Verify that a plot was successfully created
        self.assertGreater(len(plt.get_fignums()), 0)


if __name__ == "__main__":
    unittest.main()
