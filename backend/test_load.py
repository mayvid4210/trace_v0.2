from config import CACHE_DIR
import fastf1

session = fastf1.get_session(2024, "Bahrain", "R")
session.load()

print(session.laps.columns.tolist())
print(session.laps.head())