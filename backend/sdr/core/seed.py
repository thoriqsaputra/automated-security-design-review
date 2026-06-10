import logging
from sdr.core.database import SessionLocal
from sdr.apps.standards.models.base import StandardCategory

logger = logging.getLogger(__name__)

def seed():
    with SessionLocal() as db:
        categories = [
            {"code": "web_application", "name": "Web Application", "description": "Web Application Security Standard"},
        ]
        seeded_count = 0
        for cat in categories:
            existing = db.query(StandardCategory).filter(StandardCategory.code == cat["code"]).first()
            if not existing:
                new_cat = StandardCategory(**cat)
                db.add(new_cat)
                seeded_count += 1
        db.commit()
        if seeded_count > 0:
            logger.info(f"Seeded {seeded_count} standard categories.")

if __name__ == "__main__":
    seed()
