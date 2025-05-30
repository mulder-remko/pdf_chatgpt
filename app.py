import streamlit as st
import openai
import os
# wenn semantic_data.pkl fehlt,automatisch index bauen
if not os.path.exists("semantic_data.pkl"):
    from semantic_index import build_index  # du musst build_index in semantic_index.py exportieren
    build_index()
import pickle
import numpy as np
from datetime import datetime
from fpdf import FPDF
from collections import Counter
import pandas as pd
# Direkt hier deinen Key eintragen – gerade Anführungszeichen!
# openai.api_key = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")
st.write("🔑 OPENAI_API_KEY loaded:", openai.api_key is not None)
# 🔐 Benutzer eingeben
st.set_page_config(page_title="GPT-Service", layout="centered")
nutzer = st.text_input("🔐 Dein Name oder Kürzel:", value="", max_chars=20)
if not nutzer:
    st.stop()

# 🔄 Verlauf initialisieren
if "verlauf" not in st.session_state:
    st.session_state["verlauf"] = []

# GPT-Key aus Umgebungsvariable
# openai.api_key = os.getenv("OPENAI_API_KEY")

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
            st.markdown(f"❓ {frage_text} — **{anzahl}x gestellt**")
