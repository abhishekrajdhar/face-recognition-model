import os

# Distance threshold for a positive face match (lower is stricter)
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))

# Minimum seconds between two attendance marks for the same person
ATTENDANCE_COOLDOWN_SECONDS = int(os.getenv("ATTENDANCE_COOLDOWN_SECONDS", "60"))

# SQLite file path
SQLITE_URL = os.getenv("SQLITE_URL", "sqlite:///./attendance.db")


