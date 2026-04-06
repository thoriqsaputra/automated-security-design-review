from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, func, JSON
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
class TSDDocument(Base):
    __tablename__ = "tsd_documents"
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    
    nodes = relationship("DocumentNode", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("ArchitectureEntity", back_populates="document", cascade="all, delete-orphan")

class DocumentNode(Base):
    """Stores text chunks and their bottom-up summaries (RAPTOR tree)."""
    __tablename__ = "document_nodes"
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("tsd_documents.id"))
    text_content = Column(Text, nullable=False)
    layer = Column(Integer, default=0)
    parent_id = Column(String, ForeignKey("document_nodes.id"), nullable=True)
    embedding = Column(Vector(768))
    
    document = relationship("TSDDocument", back_populates="nodes")
    parent = relationship("DocumentNode", remote_side=[id], backref="children")

class ArchitectureEntity(Base):
    """Stores extracted architectural components as Graph Nodes."""
    __tablename__ = "architecture_entities"
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("tsd_documents.id"))
    name = Column(String, index=True, nullable=False)
    entity_type = Column(String)
    description = Column(Text)
    embedding = Column(Vector(768))
    spatial_metadata = Column(JSON, nullable=True)  # Stores coordinates, trust_zone, labels_on_diagram from vision extraction
    
    document = relationship("TSDDocument", back_populates="entities")
    outgoing_edges = relationship("ArchitectureEdge", foreign_keys="[ArchitectureEdge.source_id]", back_populates="source", cascade="all, delete-orphan")
    incoming_edges = relationship("ArchitectureEdge", foreign_keys="[ArchitectureEdge.target_id]", back_populates="target", cascade="all, delete-orphan")

class ArchitectureEdge(Base):
    """Stores data flows and relationships between entities as Graph Edges."""
    __tablename__ = "architecture_edges"
    id = Column(String, primary_key=True, index=True)
    source_id = Column(String, ForeignKey("architecture_entities.id"))
    target_id = Column(String, ForeignKey("architecture_entities.id"))
    relation_type = Column(String, nullable=False)
    description = Column(Text)
    
    source = relationship("ArchitectureEntity", foreign_keys=[source_id])
    target = relationship("ArchitectureEntity", foreign_keys=[target_id])