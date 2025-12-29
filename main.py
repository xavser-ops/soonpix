"""
FastAPI server for face embedding generation using InsightFace.
Deploy this on Railway, Render, Fly.io, or any Python hosting service.
"""

import base64
import io
import os
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis
from pydantic import BaseModel

app = FastAPI(
    title="Face Vectorization API",
    description="Generate 512-dimension face embeddings using InsightFace/ArcFace",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize InsightFace model
# Uses buffalo_l model which provides 512-dim embeddings
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]  # Use CUDAExecutionProvider for GPU
)
face_app.prepare(ctx_id=0, det_size=(640, 640))


class VectorizeRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None


class VectorizeResponse(BaseModel):
    success: bool
    embedding: Optional[list] = None
    face_count: int = 0
    error: Optional[str] = None
    face_bbox: Optional[list] = None


class MultiVectorizeResponse(BaseModel):
    success: bool
    faces: list = []
    face_count: int = 0
    error: Optional[str] = None


def load_image_from_url(url: str) -> np.ndarray:
    """Download and decode image from URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from URL")
    return image


def load_image_from_base64(base64_str: str) -> np.ndarray:
    """Decode image from base64 string."""
    # Remove data URL prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    image_data = base64.b64decode(base64_str)
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode base64 image")
    return image


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Face Vectorization API",
        "model": "InsightFace buffalo_l (ArcFace)",
        "embedding_dim": 512
    }


@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_face(request: VectorizeRequest):
    """
    Generate face embedding from an image.
    Returns the embedding of the first/largest face detected.
    """
    try:
        # Load image
        if request.image_url:
            image = load_image_from_url(request.image_url)
        elif request.image_base64:
            image = load_image_from_base64(request.image_base64)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either image_url or image_base64 is required"
            )
        
        # Detect faces and extract embeddings
        faces = face_app.get(image)
        
        if not faces:
            return VectorizeResponse(
                success=False,
                face_count=0,
                error="No face detected in the image"
            )
        
        # Get the largest face (most prominent)
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Extract embedding (512-dim vector)
        embedding = largest_face.embedding.tolist()
        
        # Normalize the embedding (L2 normalization for cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()
        
        return VectorizeResponse(
            success=True,
            embedding=embedding,
            face_count=len(faces),
            face_bbox=largest_face.bbox.tolist()
        )
        
    except requests.RequestException as e:
        return VectorizeResponse(
            success=False,
            error=f"Failed to download image: {str(e)}"
        )
    except Exception as e:
        return VectorizeResponse(
            success=False,
            error=f"Processing error: {str(e)}"
        )


@app.post("/vectorize-all", response_model=MultiVectorizeResponse)
async def vectorize_all_faces(request: VectorizeRequest):
    """
    Generate face embeddings for ALL faces detected in an image.
    Useful for group photos.
    """
    try:
        # Load image
        if request.image_url:
            image = load_image_from_url(request.image_url)
        elif request.image_base64:
            image = load_image_from_base64(request.image_base64)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either image_url or image_base64 is required"
            )
        
        # Detect faces and extract embeddings
        faces = face_app.get(image)
        
        if not faces:
            return MultiVectorizeResponse(
                success=False,
                face_count=0,
                error="No faces detected in the image"
            )
        
        face_data = []
        for i, face in enumerate(faces):
            # Extract and normalize embedding
            embedding = face.embedding.tolist()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
            
            face_data.append({
                "index": i,
                "embedding": embedding,
                "bbox": face.bbox.tolist(),
                "det_score": float(face.det_score) if hasattr(face, 'det_score') else None
            })
        
        return MultiVectorizeResponse(
            success=True,
            faces=face_data,
            face_count=len(faces)
        )
        
    except Exception as e:
        return MultiVectorizeResponse(
            success=False,
            error=f"Processing error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
