import enum


class ReviewStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED_CLEAN = "completed_clean"
    COMPLETED_WITH_FINDINGS = "completed_with_findings"
    FAILED = "failed"
    CANCELLED = "cancelled"
    APPROVED = "approved"
    REJECTED = "rejected"


class FindingStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class MetStatus(str, enum.Enum):
    MET = "met"
    NOT_MET = "not_met"
    NA = "na"


class Severity(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingType(str, enum.Enum):
    REQUIREMENT = "requirement"
    DIAGRAM = "diagram"


class AnchorType(str, enum.Enum):
    TEXT = "text"
    DIAGRAM = "diagram"
