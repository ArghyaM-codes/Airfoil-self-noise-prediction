import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(y_true, y_pred):
    """Calculates and prints MSE, RMSE, and R2 score for model predictions."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("--- Model Performance Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    return {"mse": mse, "rmse": rmse, "r2": r2}


def plot_predictions(y_true, y_pred, title="Actual vs Predicted Sound Pressure"):
    """Generates a scatter plot comparing actual vs predicted sound pressure levels."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color="blue")
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        lw=2,
    )
    plt.xlabel("Actual Sound Pressure Level (dB)")
    plt.ylabel("Predicted Sound Pressure Level (dB)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
