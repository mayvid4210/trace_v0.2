
import numpy as np
import pandas as pd


DATA_PATH = "data/processed/resistance_data.parquet"


def fit_curve(df):

    x = (
        df["air_density"]
        * df["Speed_ms"] ** 2
    ).to_numpy()

    y = df["resistance_force_n"].to_numpy()

    valid = (
        np.isfinite(x) &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 20:
        return None

    coefficients = np.polyfit(
        x,
        y,
        1
    )

    c1 = coefficients[0]
    c0 = coefficients[1]

    predicted = (
        c0 +
        c1 * x
    )

    residuals = y - predicted

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    mae = np.mean(
        np.abs(residuals)
    )

    rmse = np.sqrt(
        np.mean(residuals ** 2)
    )

    return {
        "samples": len(x),
        "c0": c0,
        "c1": c1,
        "r2": r2,
        "mae": mae,
        "rmse": rmse
    }


def main():

    df = pd.read_parquet(
        DATA_PATH
    )

    print(
        "\n========== RESISTANCE FIT BY SESSION =========="
    )

    for session in sorted(
        df["SessionName"].dropna().unique()
    ):

        subset = df[
            df["SessionName"] == session
        ].copy()

        result = fit_curve(
            subset
        )

        if result is None:
            print(
                f"\n{session}: insufficient data"
            )
            continue

        print(
            f"\n--- {session} ---"
        )

        print(
            f"Samples: {result['samples']}"
        )

        print(
            "F_resistance = "
            f"{result['c0']:.3f} + "
            f"{result['c1']:.8f} * rho * v²"
        )

        print(
            f"R²:   {result['r2']:.4f}"
        )

        print(
            f"MAE:  {result['mae']:.2f} N"
        )

        print(
            f"RMSE: {result['rmse']:.2f} N"
        )


if __name__ == "__main__":
    main()
