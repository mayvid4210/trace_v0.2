import pandas as pd


def build_status_intervals(session):
    ts = (
        session.track_status
        .copy()
        .sort_values("Time")
        .reset_index(drop=True)
    )

    ts["EndTime"] = ts["Time"].shift(-1)

    return ts

def lap_contamination(lap_start, lap_end, status_intervals):
    """Calculate status overlap seconds for one lap."""

    total = (lap_end - lap_start).total_seconds()

    if total <= 0 or pd.isna(lap_start) or pd.isna(lap_end):
        return {}, total

    breakdown = {}

    for _, row in status_intervals.iterrows():

        # If this is the last status interval,
        # let it run until the end of this lap.
        seg_end = (
            row["EndTime"]
            if pd.notna(row["EndTime"])
            else lap_end
        )

        overlap_start = max(lap_start, row["Time"])
        overlap_end = min(lap_end, seg_end)

        overlap = (
            overlap_end - overlap_start
        ).total_seconds()

        if overlap > 0:
            code = str(row["Status"])

            breakdown[code] = (
                breakdown.get(code, 0) + overlap
            )

    return breakdown, total

STATUS_MAP = {
    "1": "clear",
    "2": "yellow",
    "4": "safety_car",
    "5": "red",
    "6": "vsc",
    "7": "vsc",
}


def classify_lap(lap, status_intervals):

    breakdown, total = lap_contamination(
        lap["LapStartTime"],
        lap["Time"],
        status_intervals
    )

    if total <= 0:
        return {
            "category": "INVALID",
            "clean_weight": 0.0
        }

    pct = {}

    for code, seconds in breakdown.items():

        label = STATUS_MAP.get(
            str(code),
            "unknown"
        )

        pct[label] = (
            pct.get(label, 0)
            + seconds / total
        )

    pct_sc = pct.get("safety_car", 0)
    pct_vsc = pct.get("vsc", 0)
    pct_yellow = pct.get("yellow", 0)
    pct_red = pct.get("red", 0)

    contamination = (
        pct_sc
        + pct_vsc
        + pct_yellow
        + pct_red
    )

    clean_weight = round(
        max(0.0, 1 - contamination),
        3
    )

    if pct_red > 0.05:
        category = "INVALID"
        clean_weight = 0.0

    elif pct_sc >= 0.5:
        category = "SAFETY_CAR"

    elif pct_vsc >= 0.5:
        category = "VSC"

    elif pct_yellow >= 0.5:
        category = "YELLOW_FLAG"

    elif contamination > 0.05:
        category = "PARTIAL"

    else:
        category = "CLEAN"

    return {
        "category": category,
        "pct_safety_car": round(pct_sc, 3),
        "pct_vsc": round(pct_vsc, 3),
        "pct_yellow": round(pct_yellow, 3),
        "clean_weight": clean_weight
    }

def clean_laps(session, laps_df):
    intervals = build_status_intervals(session)

    results = laps_df.apply(
        lambda lap: classify_lap(lap, intervals),
        axis=1,
        result_type="expand"
    )

    cleaned = pd.concat(
        [
            laps_df.reset_index(drop=True),
            results
        ],
        axis=1
    )

    cleaned = flag_out_laps(cleaned)

    return cleaned

def flag_out_laps(laps_df):
    laps_df = laps_df.copy()

    laps_df["is_out_lap"] = laps_df["PitOutTime"].notna()

    laps_df.loc[
        laps_df["is_out_lap"],
        "category"
    ] = "OUT_LAP"

    laps_df.loc[
        laps_df["is_out_lap"],
        "clean_weight"
    ] = 0.0

    return laps_df