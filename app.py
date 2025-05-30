import streamlit as st
import openai
import os

# ➊ Page-Config ganz am Anfang
st.set_page_config(page_title="GPT-Service", layout="centered")

# ➋ DEBUG via print (erscheint im Build-Log, nicht im App-UI)
api_key = os.getenv("OPENAI_API_KEY")
print("DEBUG: OPENAI_API_KEY is set:", bool(api_key))

# ➌ OpenAI konfigurieren
openai.api_key = api_key

# ➍ Index bauen, falls nötig
if not os.path.exists("semantic_data.pkl"):
    from semantic_index import build_index
    build_index()

# Nun alle weiteren Imports
import pickle
import numpy as np
from datetime import datetime
from fpdf import FPDF
from collections import Counter
import pandas as pd

# 🔐 Benutzer eingeben
nutzer = st.text_input("🔐 Dein Name oder Kürzel:", value="", max_chars=20)
if not nutzer:
    st.stop()

# 🔄 Verlauf initialisieren
if "verlauf" not in st.session_state:
    st.session_state["verlauf"] = []

st.title("🤖 GPT-Servicefragen")
st.markdown("Frag die KI zu deinen Dokumenten.")

# 🔍 Semantische Indexdaten laden
with open("semantic_data.pkl", "rb") as f:
    index, chunks, metadaten, model = pickle.load(f)

# Eingabefeld
frage = st.text_input("❓ Deine Frage:")

# Bei Eingabe: semantisch suchen + GPT antworten
if frage:
    with st.spinner("GPT denkt nach..."):
        frage_embedding = model.encode([frage])
        _, treffer_idx = index.search(np.array(frage_embedding), 3)

        dokument_info = ""
        for i in treffer_idx[0]:
            dokument_info += f"📄 {metadaten[i]}:\n{chunks[i]}\n\n"

 
        response = openai.chat.completions.create(

            model="gpt-4",
            messages=[
                {"role": "system", "content": "Beantworte die folgende Frage auf Basis dieser Inhalte:\n" + dokument_info},
                {"role": "user", "content": frage}
            ],
            temperature=0.2,
            max_tokens=500
        )
        antwort = response['choices'][0]['message']['content']
        st.success(antwort)

        # Verlauf speichern
        st.session_state["verlauf"].append((frage, antwort))
        with open("verlauf.csv", "a", encoding="utf-8") as f:
            f.write(f'"{nutzer}";"{frage}";"{antwort}"\n')

# Verlauf anzeigen
with st.expander("🕒 Gesprächsverlauf"):
    for i, (f, a) in enumerate(st.session_state["verlauf"], 1):
        st.markdown(f"**{i}. Frage:** {f}")
        st.markdown(f"💬 {a}")
        st.markdown("---")

# PDF exportieren
if st.button("📄 Verlauf als PDF speichern"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Gesprächsverlauf", ln=1, align="L")
    for i, (f, a) in enumerate(st.session_state["verlauf"], 1):
        pdf.multi_cell(0, 10, txt=f"{i}. Frage: {f}\nAntwort: {a}\n", border=0)
    pdf.output("verlauf.pdf")
    st.success("PDF wurde gespeichert: verlauf.pdf")

# Häufigste Fragen
if os.path.exists("verlauf.csv"):
    df = pd.read_csv("verlauf.csv", sep=";", names=["Name", "Frage", "Antwort"])
    haeufigkeiten = Counter(df["Frage"])
    haeufigste = haeufigkeiten.most_common(5)
    with st.expander("🌟 Häufigste Fragen"):
        for frage_text, anzahl in haeufigste:
            st.markdown(f"❓ {frage_text} — **{anzahl}x gestellt**") hier muss das korrekt rein: import streamlit as st
import openai  # schon hier importieren

# 🔒 API-Key hartkodiert, damit er auf jeden Fall genutzt wird
openai.api_key = "sk-ABC123…dein vollständiger Key…"

import os
import pickle
import numpy as np
from datetime import datetime
from fpdf import FPDF
from collections import Counter
import pandas as pd

# Page-Config MUSS jetzt als allererstes Streamlit-Kommando stehen
st.set_page_config(page_title="GPT-Service", layout="centered")Antwort"])
    haeufigkeiten = Counter(df["Frage"])
    haeufigste = haeufigkeiten.most_common(5)
    with st.expander("🌟 Häufigste Fragen"):
        for frage_text, anzahl in haeufigste:
            st.markdown(f"❓ {frage_text} — **{anzahl}x gestellt**")
