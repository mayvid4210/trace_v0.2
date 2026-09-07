import pandas as pd


def build_track_status_intervals(session):
    status = session.track_status.copy()

    status["Time"] = pd.to_timedelta(status["Time"])

    intervals = []

    for i in range(len(status)):
        start = status.iloc[i]["Time"]
        code = status.iloc[i]["Status"]
        message = status.iloc[i]["Message"]

        # The next status change marks the end
        if i + 1 < len(status):
            end = status.iloc[i + 1]["Time"]
        else:
            end = pd.NaT

        intervals.append({
            "Start": start,
            "End": end,
            "Status": code,
            "Message": message
        })

    return pd.DataFrame(intervals)