# Serves the browser chess player (research/web_player.py): the supervised
# policy wrapped in material negamax + capture-quiescence search.
# The trained weights are baked in from artifacts/ (see .dockerignore), so the
# image is self-contained and needs no volume or download at boot.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

# PaaS hosts inject PORT; HOST=0.0.0.0 is required for the container port map.
ENV PORT=8765 \
    HOST=0.0.0.0
EXPOSE 8765

CMD ["python", "research/web_player.py"]
