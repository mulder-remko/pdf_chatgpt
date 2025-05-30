from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
from PyPDF2 import PdfReader

def build_index():
    # Modell & Ordner definieren
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pdf_ordner = "pdfs"
    chunks = []
    metadaten = []

    # Text in handhabbare Abschnitte (Chunks) zerteilen
    def chunk_text(text, max_length=500):
        return [text[i:i+max_length] for i in range(0, len(text), max_length)]

    # PDFs einlesen und in Chunks zerlegen
    for datei in os.listdir(pdf_ordner):
        if not datei.lower().endswith(".pdf"):
            continue
        try:
            reader = PdfReader(os.path.join(pdf_ordner, datei))
        except Exception as e:
            print(f"⚠️ Überspringe {datei}: {e}")
            continue
        text = "".join([seite.extract_text() or "" for seite in reader.pages])
        for abschnitt in chunk_text(text):
            chunks.append(abschnitt)
            metadaten.append(datei)

    # Embeddings berechnen und FAISS-Index befüllen
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    # Index + Chunks + Metadaten + Modell speichern
    with open("semantic_data.pkl", "wb") as f:
        pickle.dump((index, chunks, metadaten, model), f)
    print("✅ semantic_data.pkl erfolgreich erstellt.")

if __name__ == "__main__":
    # Wenn das Skript direkt ausgeführt wird, baue den Index
    build_index()
