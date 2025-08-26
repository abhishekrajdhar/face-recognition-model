import argparse
import os
import pickle
from PIL import Image
import numpy as np

from sqlalchemy.orm import Session
from app.database import SessionLocal, engine, Base
from app.models import Person
from app.face.encoder import extract_face_embedding

Base.metadata.create_all(bind=engine)

def enroll_folder(dataset_dir: str):
    db: Session = SessionLocal()
    count_ok, count_fail = 0, 0
    for entry in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(person_path):
            continue

        # Expect folder name like: U001_Abhishek (ID_Name)
        if "_" in entry:
            pid, name = entry.split("_", 1)
        else:
            pid, name = entry, None

        # Pick the first image that yields a face embedding
        emb_found = None
        for fname in os.listdir(person_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(person_path, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    emb = extract_face_embedding(np.array(img))
                    if emb is not None:
                        emb_found = emb
                        break
                except Exception as e:
                    print(f"[skip] {img_path}: {e}")
                    continue

        if emb_found is None:
            print(f"[fail] No face found for {entry}")
            count_fail += 1
            continue

        # Upsert
        person = db.query(Person).filter(Person.person_id == pid).one_or_none()
        if person is None:
            person = Person(person_id=pid, name=name, embedding=pickle.dumps(emb_found))
            db.add(person)
        else:
            if name:
                person.name = name
            person.embedding = pickle.dumps(emb_found)

        db.commit()
        print(f"[ok] Enrolled {pid} ({name})")
        count_ok += 1

    db.close()
    print(f"Done. ok={count_ok}, fail={count_fail}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset folder")
    args = parser.parse_args()
    enroll_folder(args.dataset)
