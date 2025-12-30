# 🔍 Face Vectorization API

API FastAPI pour générer des embeddings faciaux 512 dimensions avec InsightFace/ArcFace.

## 📋 Prérequis

- Python 3.10+ ou Docker
- ~2GB d'espace disque (pour les modèles InsightFace)

## 🚀 Déploiement

### Option 1: Railway (Recommandé - Simple)

1. Créez un compte sur [Railway](https://railway.app)
2. Cliquez sur "New Project" → "Deploy from GitHub repo"
3. Connectez votre repo contenant ces fichiers
4. Railway détecte automatiquement le Dockerfile
5. Copiez l'URL générée (ex: `https://face-api-xxx.railway.app`)

### Option 2: Render

1. Créez un compte sur [Render](https://render.com)
2. Nouveau → Web Service → Connectez votre repo
3. Sélectionnez "Docker" comme environnement
4. Déployez et copiez l'URL

### Option 3: Fly.io

```bash
# Installez flyctl
curl -L https://fly.io/install.sh | sh

# Connectez-vous
fly auth login

# Déployez
fly launch --name face-vectorize-api
fly deploy
```

### Option 4: Local avec Docker

```bash
# Build
docker build -t face-vectorize-api .

# Run
docker run -p 8000:8000 face-vectorize-api
```

### Option 5: Local sans Docker

```bash
# Créez un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installez les dépendances
pip install -r requirements.txt

# Lancez le serveur
python main.py
```

## 🔧 Configuration dans Lovable

Une fois l'API déployée, ajoutez le secret dans Lovable :

1. Allez dans **Settings → Secrets**
2. Ajoutez : `FACE_VECTORIZE_API_URL` = `https://votre-api.railway.app/vectorize`

## 📖 API Endpoints

### `GET /`
Health check - Vérifie que l'API fonctionne.

### `POST /vectorize`
Génère l'embedding du visage principal (le plus grand) dans l'image.

**Request:**
```json
{
  "image_url": "https://example.com/photo.jpg"
}
// ou
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**
```json
{
  "success": true,
  "embedding": [0.0234, -0.0567, ...],  // 512 valeurs
  "face_count": 1,
  "face_bbox": [x1, y1, x2, y2]
}
```

### `POST /vectorize-all`
Génère les embeddings de TOUS les visages détectés.

**Response:**
```json
{
  "success": true,
  "face_count": 3,
  "faces": [
    {
      "index": 0,
      "embedding": [...],
      "bbox": [x1, y1, x2, y2],
      "det_score": 0.98
    },
    ...
  ]
}
```

## ⚡ Performance

| Métrique | Valeur |
|----------|--------|
| Dimension embedding | 512 |
| Temps moyen (CPU) | ~500ms |
| Temps moyen (GPU) | ~50ms |
| Modèle | buffalo_l (ArcFace) |

## 🔒 Sécurité

Pour la production, ajoutez une authentification :

```python
from fastapi import Header, HTTPException

API_KEY = os.environ.get("API_KEY")

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.post("/vectorize")
async def vectorize_face(request: VectorizeRequest, api_key: str = Depends(verify_api_key)):
    ...
```

## 📝 Notes

- Les embeddings sont normalisés L2 pour la similarité cosinus
- Le modèle buffalo_l offre le meilleur équilibre précision/vitesse
- Pour GPU, changez `CPUExecutionProvider` → `CUDAExecutionProvider`
