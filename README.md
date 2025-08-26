# ML Attendance Recognition API (FastAPI + face_recognition)

A simple, production-ready starter for marking attendance from user-captured photos.
- Face enrollment (register users)
- Face recognition + attendance marking
- SQLite database via SQLAlchemy
- In-memory embedding cache for speed

## ⚠️ System Notes
- `face_recognition` depends on `dlib`. On Windows, install prebuilt wheels if pip build fails.
  Try: `pip install cmake` then `pip install dlib==19.24.2` (or use a prebuilt wheel), then `pip install face_recognition`.
- On Linux, ensure `cmake` and build tools are installed. Alternatively, use a Dockerfile (below).

## 🔧 Setup (Local)
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows
# or: source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## 📁 Project Structure
```text
app/
  main.py
  config.py
  database.py
  models.py
  schemas.py
  face/
    __init__.py
    encoder.py
scripts/
  enroll_from_folder.py
```

## ▶️ Endpoints
- `GET /health` → health check
- `POST /register` (multipart) → enroll a person
  - fields: `person_id` (str, required), `name` (str, optional), `file` (image)
- `POST /mark-attendance` (multipart) → recognize + mark attendance
  - fields: `file` (image). Returns recognized `person_id`/`name` if matched.
- `GET /attendance` → list recent attendance
- `GET /people` → list enrolled people

## 🧪 cURL Examples
```bash
# Enroll
curl -X POST "http://localhost:8000/register"       -F "person_id=U001"       -F "name=Abhishek Dubey"       -F "file=@/path/to/abhishek.jpg"

# Mark attendance
curl -X POST "http://localhost:8000/mark-attendance"       -F "file=@/path/to/capture.jpg"
```

## 🐳 Docker (optional)
```Dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential cmake libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📦 Bulk Enrollment (folder of faces)
Put images in a structure like:
```text
dataset/
  U001_Abhishek/
    img1.jpg
    img2.jpg
  U002_Priya/
    img1.jpg
```
Then run:
```bash
python scripts/enroll_from_folder.py --dataset ./dataset
```

## 🔐 Spoofing / Liveness (Future Work)
- Add liveness detection (blink detection, texture-based CNN, or phone sensor checks).
- Add geo/time rules (e.g., only mark between 9AM–10AM IST).

## 🧰 Notes
- Matching threshold defaults to 0.6 (FaceNet/face_recognition typical). Adjust via `FACE_MATCH_THRESHOLD` env var.
- One face per request is assumed. If multiple faces, the first face is used.
- Attendance marked only once per person per minute to avoid duplicates (configurable).
