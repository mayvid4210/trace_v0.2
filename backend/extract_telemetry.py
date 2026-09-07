import pandas as pd


def extract_telemetry(session, year, gp):
    frames = []

    for _, lap in session.laps.iterlaps():
        try:
            tel = lap.get_car_data().add_distance()
        except Exception:
            continue

        tel = tel[
            ["Time", "Distance", "Speed", "RPM",
             "nGear", "Throttle", "Brake"]
        ].copy()

        tel["Year"] = year
        tel["GrandPrix"] = gp
        tel["SessionName"] = session.name
        tel["Driver"] = lap["Driver"]
        tel["LapNumber"] = lap["LapNumber"]

        frames.append(tel)

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )