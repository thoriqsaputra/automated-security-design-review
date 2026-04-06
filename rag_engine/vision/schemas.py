from typing import List, Optional
from pydantic import BaseModel, Field

class ComponentCoordinates(BaseModel):
    """
    Normalized coordinates on a 0-1000 scale. 
    This is the native format Gemini uses for spatial understanding.
    """
    ymin: int = Field(..., description="Top edge (0-1000)")
    xmin: int = Field(..., description="Left edge (0-1000)")
    ymax: int = Field(..., description="Bottom edge (0-1000)")
    xmax: int = Field(..., description="Right edge (0-1000)")

class VisualComponent(BaseModel):
    name: str = Field(..., description="UPPERCASE exact component name from diagram (e.g., 'PAYMENT API')")
    type: str = Field(..., description="Component type: Frontend, Backend, Database, ExternalAPI, etc.")
    coordinates: ComponentCoordinates
    labels_on_box: List[str] = Field(default_factory=list, description="Additional text/labels visible inside or next to the box")

class VisualDataFlow(BaseModel):
    source: str = Field(..., description="Source component name (UPPERCASE). MUST exactly match a component name.")
    target: str = Field(..., description="Target component name (UPPERCASE). MUST exactly match a component name.")
    flow_label: str = Field(..., description="Text written on the arrow (e.g., 'REST API calls'). If none, use 'CONNECTS_TO'")
    protocol: Optional[str] = Field(None, description="Inferred protocol if visible (HTTPS, gRPC, etc.)")
    direction: str = Field(default="unidirectional", description="'unidirectional' or 'bidirectional'")

class TrustBoundary(BaseModel):
    boundary_name: str = Field(..., description="Name of the trust boundary (e.g., 'Production VPC', 'DMZ')")
    contained_components: List[str] = Field(..., description="List of exact component names physically located inside this boundary")
    boundary_type: str = Field(..., description="Type: Network Perimeter, Security Zone, Administrative Domain, etc.")

class ArchitectureDiagramExtraction(BaseModel):
    components: List[VisualComponent] = Field(..., description="All extracted components from the diagram")
    data_flows: List[VisualDataFlow] = Field(..., description="All extracted data flows connecting the components")
    trust_boundaries: List[TrustBoundary] = Field(default_factory=list, description="All extracted trust boundaries grouping the components")