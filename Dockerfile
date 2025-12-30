# Utilisation de Python 3.11 (Version stable et compatible)
FROM python:3.11-slim

# 1. Installation des dépendances système (Critique pour OpenCV et InsightFace)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Définition du dossier de travail
WORKDIR /app

# 2. Copie des requirements d'abord (pour le cache Docker)
COPY requirements.txt .

# 3. Installation des librairies Python
# On ajoute --no-cache-dir pour alléger l'image
RUN pip install --no-cache-dir -r requirements.txt

# 4. Téléchargement anticipé des modèles InsightFace
# Cela évite le crash au premier démarrage
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0)"

# 5. COPIE DE TOUT LE RESTE (C'est ici que main.py sera copié)
COPY . .

# 6. Exposition du port
EXPOSE 8000

# 7. Lancement de l'API avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
