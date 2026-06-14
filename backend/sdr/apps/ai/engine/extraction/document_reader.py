from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, Optional


class StandardDocumentReader:
    def __init__(
        self,
        *,
        get_local_file_path: Callable[[Any], AbstractContextManager],
        get_document_content: Callable[..., Dict[str, Any]],
    ) -> None:
        self._get_local_file_path = get_local_file_path
        self._get_document_content = get_document_content

    def read_source_document(
        self,
        source_doc,
        *,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self._get_local_file_path(source_doc.document) as source_doc_path:
            return self._get_document_content(
                source_doc_path,
                source_doc.document,
                start_page=start_page,
                end_page=end_page,
            )
