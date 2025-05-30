
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
from PyPDF2 import PdfReader

model = SentenceTransformer("all-MiniLM-L6-v2")
pdf_ordner = "pdfs"
chunks = []
metadaten = []

def chunk_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

for datei in os.listdir(pdf_ordner):
    if datei.endswith(".pdf"):
        reader = PdfReader(os.path.join(pdf_ordner, datei))
        text = "".join([seite.extract_text() or "" for seite in reader.pages])
        for abschnitt in chunk_text(text):
            chunks.append(abschnitt)
            metadaten.append(datei)

# Vektorisieren
embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

# Speichern
with open("semantic_data.pkl", "wb") as f:
    pickle.dump((index, chunks, metadaten, model), f)
