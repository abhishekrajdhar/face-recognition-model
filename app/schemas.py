from pydantic import BaseModel
from datetime import datetime

class PersonOut(BaseModel):
    person_id: str
    name: str | None = None

    class Config:
        from_attributes = True

class AttendanceOut(BaseModel):
    id: int
    person_id: str
    name: str | None = None
    timestamp: datetime
    source: str | None = None
