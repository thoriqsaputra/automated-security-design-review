from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, func
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector
import uuid

Base = declarative_base()

class SecurityRequirement(Base):
    """
    Adjacency List Graph for storing Hierarchical Security Requirements.
    This replaces raw ingestion noise with pure actionable rules (Nodes/Edges).
    """
    __tablename__ = 'security_requirements'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, index=True, nullable=False)
    category = Column(String, index=True, nullable=False) # e.g. "OWASP", "NIST"
    title = Column(String, nullable=False)
    actionable_rule = Column(Text, nullable=False)
    
    parent_id = Column(String, ForeignKey("security_requirements.id", ondelete="CASCADE"), nullable=True)
    
    embedding = Column(Vector(768), nullable=True)
    
    # Back-population traversal link
    children = relationship(
        "SecurityRequirement", 
        backref="parent", 
        remote_side=[id],
        cascade="all, delete-orphan", 
        single_parent=True
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
