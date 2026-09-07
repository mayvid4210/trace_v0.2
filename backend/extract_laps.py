import pandas as pd


def extract_laps(session, year, gp):
    laps = session.laps.copy()

    laps["Year"] = year
    laps["GrandPrix"] = gp
    laps["SessionName"] = session.name
    laps["Circuit"] = session.event["Location"]

    # Convert timedeltas to seconds
    time_cols = [
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "PitInTime",
        "PitOutTime",
    ]

    for col in time_cols:
        if col in laps.columns:
            laps[col + "Seconds"] = laps[col].dt.total_seconds()

    keep = [
        "Year",
        "GrandPrix",
        "Circuit",
        "SessionName",
        "Driver",
        "Team",
        "LapStartTime",
        "Time",
        "LapNumber",
        "Stint",
        "LapTime",
        "LapTimeSeconds",
        "Sector1Time",
        "Sector1TimeSeconds",
        "Sector2Time",
        "Sector2TimeSeconds",
        "Sector3Time",
        "Sector3TimeSeconds",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "PitInTime",
        "PitInTimeSeconds",
        "PitOutTime",
        "PitOutTimeSeconds",
        "TrackStatus",
        "Position",
        "SpeedI1",
        "SpeedI2",
        "SpeedFL",
        "SpeedST",
    ]

    # Keep only columns that actually exist
    keep = [c for c in keep if c in laps.columns]

    return laps[keep]