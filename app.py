
# Customized NFL Q&A App using Streamlit

import streamlit as st
import chromadb
from transformers import pipeline

def setup_documents():
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")

    # Load external text documents
    doc_files = ["data/doc1.txt", "data/doc2.txt", "data/doc3.txt", "data/doc4.txt", "data/doc5.txt"]
    my_documents = []
    for file_path in doc_files:
        with open(file_path, "r", encoding="utf-8") as f:
            my_documents.append(f.read())

    collection.add(
        documents=my_documents,
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
    return collection

def get_answer(collection, question):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]

    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic in my documents."

    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""

    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    return response[0]['generated_text'].strip()

# MAIN APP UI

st.title("🏈 NFL Knowledge Hub")
st.markdown("*Your personal football assistant*")

st.write("Welcome to my personalized NFL database! Ask me anything about football history, rules, rivalries, and stats.")

collection = setup_documents()

question = st.text_input("What would you like to know about the NFL?")

if st.button("Find My Answer", type="primary"):
    if question:
        with st.spinner("🔎 Searching my football playbook..."):
            answer = get_answer(collection, question)
        st.write("**Answer:**")
        st.write(answer)
    else:
        st.write("Please enter a question!")

with st.expander("About this NFL Q&A System"):
    st.write("""
    I created this Q&A system with documents about:
    - NFL history and league structure 🏛️
    - Player positions and record holders 🏅
    - Season format, playoffs, and international games 🌍
    - Top teams and rivalries 🔥
    - Rules, scoring, and notable records 📊

    Try asking things like:
    - How many teams are in the NFL?
    - What does a linebacker do?
    - Who has the most Super Bowl wins?
    - What is the scoring system in football?
    """)
