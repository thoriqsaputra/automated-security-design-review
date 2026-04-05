from sqlalchemy.orm import Session
from db.models import DocumentNode, ArchitectureEntity
from typing import List
import logging

# Import LlamaIndex base classes
from llama_index.core import Settings
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

logger = logging.getLogger(__name__)

class HybridThreatRetriever(BaseRetriever):
    """
    A custom LlamaIndex Retriever that executes Hybrid RAG against pgvector database.
    It combines RAPTOR semantic search with GraphRAG 1-hop traversal.
    """
    def __init__(self, db: Session, doc_id: str, top_k: int = 5):
        super().__init__()
        self.db = db
        self.doc_id = doc_id
        self.top_k = top_k
        
    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        """
        This is the required method for a LlamaIndex custom retriever.
        query_bundle.query_str contains the user's threat query.
        """
        threat_query = query_bundle.query_str
        
        try:
            query_vector = Settings.embed_model.get_query_embedding(threat_query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []
            
        vector_nodes = self.db.query(DocumentNode).filter(
            DocumentNode.document_id == self.doc_id
        ).order_by(
            DocumentNode.embedding.cosine_distance(query_vector)
        ).limit(self.top_k).all()
        
        entities = self.db.query(ArchitectureEntity).filter(
            ArchitectureEntity.document_id == self.doc_id
        ).order_by(
            ArchitectureEntity.embedding.cosine_distance(query_vector)
        ).limit(3).all()
        
        graph_context = []
        for entity in entities:
            for edge in entity.outgoing_edges:
                graph_context.append(f"[{entity.name}] --({edge.relation_type})--> [{edge.target.name}]: {edge.description}")
            for edge in entity.incoming_edges:
                graph_context.append(f"[{edge.source.name}] --({edge.relation_type})--> [{entity.name}]: {edge.description}")
                
        results = []
        
        for node in vector_nodes:
            results.append(NodeWithScore(node=TextNode(text=node.text_content), score=1.0))
            
        if graph_context:
            unique_edges = list(set(graph_context))
            topology_text = "Architecture Topology Context:\n" + "\n".join(unique_edges)
            results.append(NodeWithScore(node=TextNode(text=topology_text), score=1.0))
            
        return results