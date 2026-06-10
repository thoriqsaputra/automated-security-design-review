import os
import io
import logging
import tempfile
from typing import Optional, BinaryIO
from minio import Minio
from minio.error import S3Error
from sdr.core.config import settings

logger = logging.getLogger(__name__)

class MinioStorageService:
    def __init__(self):
        try:
            self.client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            self.bucket_name = settings.MINIO_BUCKET_NAME
            self._ensure_bucket_exists()
        except Exception as e:
            logger.error("Failed to initialize MinIO client: %s", e)
            self.client = None

    def _ensure_bucket_exists(self):
        """Ensure the target bucket exists, creating it if necessary."""
        if not self.client:
            return
        try:
            found = self.client.bucket_exists(self.bucket_name)
            if not found:
                self.client.make_bucket(self.bucket_name)
                logger.info("Created MinIO bucket '%s'", self.bucket_name)
            else:
                logger.debug("MinIO bucket '%s' already exists", self.bucket_name)
        except S3Error as e:
            logger.error("Error checking/creating bucket '%s': %s", self.bucket_name, e)

    def upload_file(self, file_data: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        """Upload a file as bytes to MinIO and return the object name."""
        if not self.client:
            raise RuntimeError("MinIO client is not initialized.")
            
        data_stream = io.BytesIO(file_data)
        length = len(file_data)
        
        try:
            self.client.put_object(
                self.bucket_name,
                object_name,
                data_stream,
                length,
                content_type=content_type,
            )
            logger.info("Successfully uploaded %s to MinIO bucket %s", object_name, self.bucket_name)
            return object_name
        except S3Error as e:
            logger.error("Failed to upload %s to MinIO: %s", object_name, e)
            raise RuntimeError(f"Storage upload failed: {e}")

    def download_to_file(self, object_name: str, file_path: str):
        """Download an object from MinIO directly to a local file path."""
        if not self.client:
            raise RuntimeError("MinIO client is not initialized.")
        try:
            self.client.fget_object(self.bucket_name, object_name, file_path)
            logger.info("Successfully downloaded %s to %s", object_name, file_path)
        except S3Error as e:
            logger.error("Failed to download %s from MinIO: %s", object_name, e)
            raise RuntimeError(f"Storage download failed: {e}")

    def delete_file(self, object_name: str):
        """Delete an object from MinIO."""
        if not self.client:
            raise RuntimeError("MinIO client is not initialized.")
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info("Successfully deleted %s from MinIO", object_name)
        except S3Error as e:
            logger.error("Failed to delete %s from MinIO: %s", object_name, e)
            raise RuntimeError(f"Storage delete failed: {e}")

# Expose a singleton instance
storage_service = MinioStorageService()
