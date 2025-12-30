import base64
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du modèle (CPU)
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
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

def load_image_from_url(url: str) -> np.ndarray:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from URL")
    return image

def load_image_from_base64(base64_str: str) -> np.ndarray:
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
    return {"status": "healthy", "service": "Face Vectorization API"}

@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_face(request: VectorizeRequest):
    try:
        if request.image_url:
            image = load_image_from_url(request.image_url)
        elif request.image_base64:
            image = load_image_from_base64(request.image_base64)
        else:
            raise HTTPException(status_code=400, detail="Image URL or Base64 required")
        
        faces = face_app.get(image)
        
        if not faces:
            return VectorizeResponse(success=False, error="No face detected")
        
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = largest_face.embedding.tolist()
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()
        
        return VectorizeResponse(
            success=True,
            embedding=embedding,
            face_count=len(faces),
            face_bbox=largest_face.bbox.tolist()
        )
        
    except Exception as e:
        return VectorizeResponse(success=False, error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
