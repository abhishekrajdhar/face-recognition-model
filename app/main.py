from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import io, pickle, time
from PIL import Image
import numpy as np

from .database import Base, engine, get_db
from .models import Person, Attendance
from .schemas import AttendanceOut, PersonOut
from .face.encoder import extract_face_embedding, compute_distance
from .config import FACE_MATCH_THRESHOLD, ATTENDANCE_COOLDOWN_SECONDS

app = FastAPI(title="ML Attendance Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables
Base.metadata.create_all(bind=engine)

# In-memory cache of embeddings to speed up matching
# key: db person id (int), value: (person_id (str), name (str|None), embedding (np.ndarray))
EMBEDDING_CACHE: Dict[int, tuple[str, Optional[str], np.ndarray]] = {}

def _load_cache(db: Session):
    EMBEDDING_CACHE.clear()
    people = db.query(Person).all()
    for p in people:
        emb = pickle.loads(p.embedding)
        EMBEDDING_CACHE[p.id] = (p.person_id, p.name, emb)

# load cache at startup
@app.on_event("startup")
def startup_event():
    db = next(get_db())
    _load_cache(db)

@app.get("/health")
def health():
    return {"status": "ok", "people_cached": len(EMBEDDING_CACHE)}

def _pil_to_np(file: UploadFile) -> np.ndarray:
    image = Image.open(file.file).convert("RGB")
    return np.array(image)

@app.post("/register")
async def register(
    person_id: str = Form(...),
    name: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Convert image to embedding
    img_np = _pil_to_np(file)
    emb = extract_face_embedding(img_np)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected in image.")

    # Upsert person
    person = db.query(Person).filter(Person.person_id == person_id).one_or_none()
    if person is None:
        person = Person(
            person_id=person_id,
            name=name,
            embedding=pickle.dumps(emb),
        )
        db.add(person)
        db.commit()
        db.refresh(person)
    else:
        # Update name and embedding if provided
        if name is not None:
            person.name = name
        person.embedding = pickle.dumps(emb)
        db.commit()
        db.refresh(person)

    # Update cache
    EMBEDDING_CACHE[person.id] = (person.person_id, person.name, emb)
    return {"status": "success", "person_id": person.person_id, "name": person.name}

_last_mark: Dict[int, float] = {}  # person_db_id -> last timestamp

@app.post("/mark-attendance")
async def mark_attendance(
    file: UploadFile = File(...),
    source: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    if len(EMBEDDING_CACHE) == 0:
        raise HTTPException(status_code=400, detail="No enrolled faces yet. Register at least one person.")

    img_np = _pil_to_np(file)
    emb = extract_face_embedding(img_np)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected in image.")

    # Find best match
    best_id = None
    best_dist = 10.0
    for db_id, (pid, pname, pemb) in EMBEDDING_CACHE.items():
        d = compute_distance(emb, pemb)
        if d < best_dist:
            best_dist = d
            best_id = db_id

    if best_id is None or best_dist > FACE_MATCH_THRESHOLD:
        return {"status": "failed", "reason": "Face not recognized", "distance": best_dist}

    # Cooldown to prevent duplicate rapid marks
    now = time.time()
    last = _last_mark.get(best_id, 0)
    if now - last < ATTENDANCE_COOLDOWN_SECONDS:
        person_id, person_name, _ = EMBEDDING_CACHE[best_id]
        return {
            "status": "success",
            "message": "Already marked recently (cooldown)",
            "person_id": person_id,
            "name": person_name,
            "distance": best_dist,
        }

    # Persist attendance
    att = Attendance(person_id=best_id, source=source)
    db.add(att)
    db.commit()
    db.refresh(att)

    _last_mark[best_id] = now

    person_id, person_name, _ = EMBEDDING_CACHE[best_id]
    return {
        "status": "success",
        "person_id": person_id,
        "name": person_name,
        "attendance_id": att.id,
        "timestamp": att.timestamp,
        "distance": best_dist,
    }

@app.get("/people", response_model=List[PersonOut])
def list_people(db: Session = Depends(get_db)):
    people = db.query(Person).all()
    return [PersonOut(person_id=p.person_id, name=p.name) for p in people]

@app.get("/attendance", response_model=List[AttendanceOut])
def list_attendance(db: Session = Depends(get_db)):
    q = (
        db.query(Attendance, Person)
        .join(Person, Attendance.person_id == Person.id)
        .order_by(Attendance.timestamp.desc())
        .all()
    )
    out = []
    for att, person in q:
        out.append(AttendanceOut(
            id=att.id,
            person_id=person.person_id,
            name=person.name,
            timestamp=att.timestamp,
            source=att.source,
        ))
    return out
