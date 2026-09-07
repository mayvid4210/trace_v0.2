import fastf1

session = fastf1.get_session(2024, "Bahrain", "R")
session.load()

print("\n=== TRACK STATUS ===")
print(session.track_status.to_string(index=False))