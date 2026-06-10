import os
import mimetypes
import urllib.request
import urllib.error
import logging

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse, FileResponse
from sdr.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents")


@router.get("/proxy")
def proxy_pdf(request: Request, url: str = Query(..., description="URL of the PDF to proxy")):
    """
    Proxy PDF requests to avoid CORS/SSL issues.
    """
    try:
        target_url = url
        if 'localhost/api/' in url:
            target_url = url.replace('https://localhost/api/', 'http://localhost:8000/api/')
            target_url = target_url.replace('http://localhost/api/', 'http://localhost:8000/api/')
        
        logger.info(f"Proxying PDF from: {url} to {target_url}")

        req = urllib.request.Request(
            target_url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        # Pass through the authorization header if it exists
        auth_header = request.headers.get('Authorization')
        if auth_header:
            req.add_header('Authorization', auth_header)

        remote_response = urllib.request.urlopen(req)
        content = remote_response.read()
        
        content_type = remote_response.headers.get('Content-Type', 'application/pdf')
        
        response = Response(content=content, media_type=content_type)
        
        content_disposition = remote_response.headers.get('Content-Disposition')
        if content_disposition:
            response.headers['Content-Disposition'] = content_disposition
        
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['X-Frame-Options'] = 'ALLOWALL'
        response.headers['Content-Security-Policy'] = "frame-ancestors 'self' *"
        
        return response

    except Exception as e:
        logger.error(f"PDF Proxy error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/serve/{file_path:path}")
def serve_pdf(file_path: str):
    """
    Serve PDF files from MEDIA_ROOT with proper headers
    to avoid browser blocking.
    """
    try:
        logger.info(f"PDF request for file_path: {file_path}")
        
        if file_path.startswith('/'):
            file_path = file_path[1:]
        
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        logger.info(f"Full path: {full_path}")
        logger.info(f"MEDIA_ROOT: {settings.MEDIA_ROOT}")
        
        # Security check - ensure the file is within MEDIA_ROOT
        if not os.path.exists(full_path):
            logger.error(f"File not found: {full_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        if not os.path.abspath(full_path).startswith(os.path.abspath(settings.MEDIA_ROOT)):
            logger.error(f"Security violation: {full_path} not in {settings.MEDIA_ROOT}")
            raise HTTPException(status_code=404, detail="File not found")
        
        content_type, _ = mimetypes.guess_type(full_path)
        if not content_type:
            content_type = 'application/pdf'

        logger.info(f"Serving document: {full_path}, type: {content_type}")
        
        headers = {
            'Content-Disposition': f'inline; filename="{os.path.basename(full_path)}"',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'ALLOWALL',
            'Cache-Control': 'public, max-age=3600',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        }

        return FileResponse(
            full_path, 
            media_type=content_type, 
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Error serving PDF file {file_path}: {str(e)}")
        raise HTTPException(status_code=404, detail="File not found")