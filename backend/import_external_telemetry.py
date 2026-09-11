import json
from pathlib import Path
import pandas as pd


try:
    from config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
except ModuleNotFoundError:
    from backend.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

RAW_DIR = EXTERNAL_DATA_DIR
OUTPUT_FILE = PROCESSED_DATA_DIR / "external_telemetry.parquet"


GP_MAP = {
    "Bahrain Grand Prix": "Bahrain",
    "Canadian Grand Prix": "Canada",
}

SESSION_MAP = {
    "Practice 1": "Practice 1",
    "Practice 2": "Practice 2",
    "Practice 3": "Practice 3",
    "Race": "Race",
}


def load_tel_file(file_path, year, gp, session_name, driver, lap_number):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tel = data.get("tel", {})

        if not tel:
            return None

        required = [
            "time",
            "rpm",
            "speed",
            "gear",
            "throttle",
            "brake",
            "drs",
            "distance",
        ]

        missing = [col for col in required if col not in tel]

        if missing:
            print(f"Skipping {file_path}: missing {missing}")
            return None

        df = pd.DataFrame(tel)

        df = df.rename(
            columns={
                "time": "Time",
                "distance": "Distance",
                "speed": "Speed",
                "rpm": "RPM",
                "gear": "nGear",
                "throttle": "Throttle",
                "brake": "Brake",
                "drs": "DRS",
            }
        )

        df = df[
            [
                "Time",
                "Distance",
                "Speed",
                "RPM",
                "nGear",
                "Throttle",
                "Brake",
                "DRS",
            ]
        ].copy()

        df["Year"] = year
        df["GrandPrix"] = gp
        df["SessionName"] = session_name
        df["Driver"] = driver
        df["LapNumber"] = lap_number

        return df

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def import_external_telemetry():
    frames = []

    for gp_folder, gp_name in GP_MAP.items():

        gp_path = RAW_DIR / gp_folder

        if not gp_path.exists():
            print(f"Missing GP folder: {gp_path}")
            continue

        for session_folder, session_name in SESSION_MAP.items():

            session_path = gp_path / session_folder

            if not session_path.exists():
                print(f"Missing session: {session_path}")
                continue

            for driver_path in session_path.iterdir():

                if not driver_path.is_dir():
                    continue

                driver = driver_path.name

                for tel_file in driver_path.glob("*_tel.json"):

                    try:
                        lap_number = int(tel_file.stem.split("_")[0])
                    except ValueError:
                        continue

                    df = load_tel_file(
                        tel_file,
                        2024,
                        gp_name,
                        session_name,
                        driver,
                        lap_number,
                    )

                    if df is not None:
                        frames.append(df)

    if not frames:
        print("No telemetry files were found.")
        return

    final_df = pd.concat(frames, ignore_index=True)

    final_df = final_df.sort_values(
        [
            "Year",
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber",
            "Time",
        ]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_parquet(OUTPUT_FILE, index=False)

    print("\nImport complete.")
    print(f"Rows: {len(final_df):,}")
    print(f"Columns: {final_df.columns.tolist()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    import_external_telemetry()