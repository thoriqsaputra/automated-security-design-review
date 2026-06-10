from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

class StandardsBigIntBase:
    """
    Abstract base class for Standards models providing common timestamp fields.
    """
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


from sdr.core.database import Base

class StandardCategory(Base, StandardsBigIntBase):
    __tablename__ = "standards_standardcategory"

    CODE_WEB_APPLICATION = "web_application"
    CODE_MOBILE = "mobile"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    def __str__(self):
        return self.name
