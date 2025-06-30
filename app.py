
# Customized NFL Q&A App using Streamlit

# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

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
    my_documents = [
        "The National Football League (NFL) was founded in 1920 as the American Professional Football Association and adopted its current name in 1922. Today, the NFL consists of 32 teams divided equally between the American Football Conference (AFC) and the National Football Conference (NFC). Each conference has four divisions: North, South, East, and West. A key historical milestone was the 1970 AFL-NFL merger, which helped solidify the NFL’s dominance in American sports. The league’s championship game, the Super Bowl, is now one of the most-watched sporting events globally. The Green Bay Packers hold the record for the most total championships (13), while the New England Patriots and Pittsburgh Steelers share the most Super Bowl wins with six each. The Arizona Cardinals, originally founded as the Morgan Athletic Club in 1898, are the oldest professional football team still active. The NFL operates under a salary cap and revenue-sharing model to maintain competitive balance. Headquarters are in New York City, and the league is governed by a Commissioner — currently Roger Goodell. The NFL’s season structure and rules have evolved, but its popularity has steadily grown into a multi-billion dollar global enterprise.", 
        "NFL teams are divided into offense, defense, and special teams, with each position serving a unique function. The quarterback (QB) leads the offense — Tom Brady, the all-time leader in passing yards and touchdowns, is widely considered the greatest QB in history. Running backs (RBs) carry the ball and catch passes; Emmitt Smith holds the record for most career rushing yards. Wide receivers (WRs) specialize in catching passes — Jerry Rice dominates the record books with the most receiving yards and touchdowns. On the offensive line (OL), players like centers and tackles protect the QB and create running lanes. On defense, players include linebackers (LBs) like Ray Lewis, known for tackles and leadership, and defensive backs such as cornerbacks (CBs) and safeties (S) who defend against the pass. Defensive linemen (DL) rush the passer and stop the run — Bruce Smith leads all-time in sacks. Special teams feature kickers and punters; Adam Vinatieri holds the record for most points scored. Every position requires specialized skills, and elite players at each have helped define eras of NFL history through their individual contributions and records.", 
        "The NFL season kicks off in early September and ends in February with the Super Bowl. Each team plays 17 regular-season games across 18 weeks. After the regular season, 14 teams (7 from each conference) enter the playoffs, culminating in the AFC and NFC Championship Games. Winners of these face off in the Super Bowl — the biggest event in American sports. The Super Bowl MVP award has been won multiple times by legends like Tom Brady, who also holds the record for most Super Bowl victories (7). Beyond the U.S., the NFL has expanded its presence internationally. The NFL International Series includes games in London, Munich, and previously Mexico City. In 2024, five international games were played, highlighting the league’s global appeal. The Jacksonville Jaguars have become a frequent team abroad, even playing back-to-back games in London. The Pro Bowl, an all-star game, is now part of “Pro Bowl Games,” held the week before the Super Bowl. NFL games are broadcast in over 180 countries, with an estimated global fanbase exceeding 300 million. The season’s mix of tradition, spectacle, and expansion continues to fuel the NFL’s popularity worldwide.", 
        "The NFL has 32 franchises with rich histories and iconic rivalries. The Dallas Cowboys are often referred to as “America’s Team” and have five Super Bowl wins. The Green Bay Packers, owned by fans, boast the most championships overall (13), while the New England Patriots lead the Super Bowl era with 6 titles and 11 appearances. The Pittsburgh Steelers also have 6 Super Bowl wins, tied with New England. Historic rivalries shape the league’s identity. The Chicago Bears vs. Green Bay Packers is the oldest and most-played rivalry, dating back to 1921 — as of 2024, the Packers lead the all-time series. The Cowboys and Eagles share a fierce NFC East rivalry with playoff implications nearly every season. In the AFC, the Steelers-Ravens rivalry is known for its physicality and postseason showdowns. West Coast clashes like 49ers vs. Seahawks have grown intense in recent decades. Each franchise has its legends, such as Joe Montana (49ers), Walter Payton (Bears), and Peyton Manning (Colts/Broncos). These rivalries reflect both competitive spirit and regional pride, drawing millions of viewers every season and fueling some of the NFL’s most memorable moments.", 
        "NFL games are played in four 15-minute quarters, with a halftime after the second quarter. Teams score points in various ways: a touchdown earns 6 points, with an option for a 1-point kick or 2-point conversion. Field goals are worth 3 points, and a safety gives 2 points to the defense. Overtime rules differ between regular season and playoffs, with each team guaranteed a possession in postseason OT unless the first drive ends in a touchdown. The longest field goal in NFL history was a 66-yarder by Justin Tucker in 2021. High-scoring games also make headlines — the 1966 game between Washington and the New York Giants totaled 113 points, still an NFL record. The game is played with 11 players per side, and penalties can drastically affect outcomes — common infractions include holding, offsides, and pass interference. Replay reviews and challenges are used to ensure fairness. In recent years, rules have evolved to protect quarterbacks and minimize head injuries. Understanding the scoring system and basic rules is essential for enjoying NFL games and analyzing the strategies behind each team’s play."
    ]
   
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
