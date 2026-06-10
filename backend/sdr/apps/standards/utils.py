import hashlib
import logging
from difflib import SequenceMatcher
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from .models import StandardCategory

logger = logging.getLogger(__name__)


def normalize_requirement_text(value: str) -> str:
    return ' '.join((value or '').strip().lower().split())


def build_parameter_analysis_text(parameter_or_text, details: Optional[str] = None) -> str:
    """
    Build the full semantic text used for retrieval and TSD analysis.
    Accepts either a CategoryParameterChild-like object or explicit text/details.
    """
    if isinstance(parameter_or_text, str):
        heading = parameter_or_text
        detail_text = details or ""
    else:
        heading = getattr(parameter_or_text, "requirement_text", "") or ""
        detail_text = getattr(parameter_or_text, "details", "") or ""

    parts = [part.strip() for part in (heading, detail_text) if part and part.strip()]
    return "\n\n".join(parts)


def stable_key(value: str) -> str:
    normalized = normalize_requirement_text(value)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:24]


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _category_keyword_map():
    return {
        StandardCategory.CODE_WEB_APPLICATION: {
            'web', 'website', 'browser', 'http', 'https', 'api', 'cookie', 'csrf',
            'xss', 'sql injection', 'session', 'frontend', 'backend', 'rest', 'graphql',
        },
        StandardCategory.CODE_MOBILE: {
            'mobile', 'android', 'ios', 'app store', 'play store', 'device', 'biometric',
            'jailbreak', 'rooted', 'apk', 'ipa', 'deep link', 'keychain', 'keystore',
        },
    }


def infer_category_code(filename: str, requirements: Optional[List[str]] = None) -> str:
    corpus_parts = [filename or '']
    if requirements:
        corpus_parts.extend(requirements)
    corpus = normalize_requirement_text(' '.join(corpus_parts))
    keyword_map = _category_keyword_map()
    scores = {code: 0 for code in keyword_map.keys()}

    for code, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in corpus:
                scores[code] += 1

    best_code = max(scores, key=scores.get)
    if scores[best_code] == 0:
        return StandardCategory.CODE_WEB_APPLICATION
    return best_code


def get_category(category_code: str, db: Session) -> Optional[StandardCategory]:
    """
    Returns the active StandardCategory for *category_code*, or None.
    """
    logger.debug("Looking up category for code: '%s'", category_code)
    return db.execute(
        select(StandardCategory)
        .where(StandardCategory.code == category_code, StandardCategory.is_active == True)
    ).scalars().first()
