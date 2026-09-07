import fastf1
import pandas as pd

from config import EVENTS
from extract_laps import extract_laps
from extract_telemetry import extract_telemetry
from extract_other import extract_weather, extract_race_control
from cleaner import clean_laps

all_laps = []
all_telemetry = []
all_weather = []
all_rc = []


for event in EVENTS:
    for sess_name in event["sessions"]:

        print(
            f"Loading {event['gp']} "
            f"{event['year']} {sess_name}..."
        )

        try:
            session = fastf1.get_session(
                event["year"],
                event["gp"],
                sess_name
            )

            session.load()

        except Exception as e:
            print(f"  Skipped ({e})")
            continue

        print("  Extracting laps...")

        laps = extract_laps(
            session,
            event["year"],
            event["gp"]
        )

        print("  Cleaning laps...")

        laps = clean_laps(
            session,
            laps
        )

        all_laps.append(laps)

        print("  Extracting telemetry...")
        all_telemetry.append(
            extract_telemetry(
                session,
                event["year"],
                event["gp"]
            )
        )

        print("  Extracting weather...")
        all_weather.append(
            extract_weather(
                session,
                event["year"],
                event["gp"]
            )
        )

        print("  Extracting race control...")
        all_rc.append(
            extract_race_control(
                session,
                event["year"],
                event["gp"]
            )
        )


print("\nSaving files...")

pd.concat(
    all_laps,
    ignore_index=True
).to_parquet(
    "data/processed/laps.parquet"
)

pd.concat(
    all_telemetry,
    ignore_index=True
).to_parquet(
    "data/processed/telemetry.parquet"
)

pd.concat(
    all_weather,
    ignore_index=True
).to_parquet(
    "data/processed/weather.parquet"
)

pd.concat(
    all_rc,
    ignore_index=True
).to_parquet(
    "data/processed/race_control.parquet"
)

print("Done. Files written to data/processed/")