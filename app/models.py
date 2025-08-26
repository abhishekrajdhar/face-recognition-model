from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, func, UniqueConstraint, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

class Person(Base):
    __tablename__ = "people"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, unique=True, index=True, nullable=False)  # external id
    name = Column(String, nullable=True)
    embedding = Column(LargeBinary, nullable=False)  # pickle.dumps(np.ndarray)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    attendances = relationship("Attendance", back_populates="person")

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("people.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    source = Column(String, nullable=True)  # optional (camera id, app, etc.)

    person = relationship("Person", back_populates="attendances")
