# 1. Baserad på en lätt Python-bild
FROM python:3.11

# 2. Skapa arbetskatalog
WORKDIR /app

# 3. Kopiera in kod och requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Kopiera resten av projektet
COPY . .

# 5. Kör RAG-session-appen
CMD ["python", "src/rag_chat_session.py"]
