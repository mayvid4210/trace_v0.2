def extract_weather(session, year, gp):
    weather = session.weather_data.copy()

    weather["Year"] = year
    weather["GrandPrix"] = gp
    weather["SessionName"] = session.name

    return weather


def extract_race_control(session, year, gp):
    rc = session.race_control_messages.copy()

    rc["Year"] = year
    rc["GrandPrix"] = gp
    rc["SessionName"] = session.name

    return rc