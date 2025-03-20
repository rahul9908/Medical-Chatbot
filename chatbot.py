import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import spacy
from neo4j import GraphDatabase

# ✅ Load GPT-2 Model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Fix attention mask issue
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

def query_gpt2(user_query):
    """
    Use GPT-2 to generate responses for medical queries.
    """
    input_text = f"Answer this medical question: {user_query}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_length=100, 
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# ✅ Neo4j Connection
NEO4J_URI = "neo4j+ssc://2ea392ee.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "IL3QMjUSdk-V_hJgZMQCYvFvGE-mZkYslCFsaUoDWxI"

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()
    
    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

def get_all_diseases_from_neo4j():
    cypher_query = """
    MATCH (d:Disease) RETURN d.name AS disease
    """
    result = neo4j_conn.run_query(cypher_query)
    return [record["disease"] for record in result]

def extract_disease_from_query(user_query):
    doc = nlp(user_query)
    all_diseases = get_all_diseases_from_neo4j()
    detected_diseases = [token.text for token in doc if token.text.lower() in [d.lower() for d in all_diseases]]
    return detected_diseases[0] if detected_diseases else None

def query_neo4j(user_query):
    cypher_query = """
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE d.name = $query
    RETURN d.name AS disease, collect(s.name) AS symptoms
    """
    result = neo4j_conn.run_query(cypher_query, {"query": user_query})
    
    if result and len(result) > 0 and "disease" in result[0] and "symptoms" in result[0]:
        disease = result[0]["disease"]
        symptoms = ", ".join(result[0]["symptoms"]) if result[0]["symptoms"] else "No symptoms available"
        return f"The disease '{disease}' has symptoms: {symptoms}."
    
    return None

def chatbot_response(user_query):
    extracted_disease = extract_disease_from_query(user_query)
    
    if extracted_disease:
        neo4j_response = query_neo4j(extracted_disease)
        if neo4j_response:
            return f"✅ Fact-based Answer (Neo4j): {neo4j_response}"
    
    llm_response = query_gpt2(user_query)
    return f"⚠️ AI-Generated Answer (GPT-2, may hallucinate): {llm_response}"

# ✅ Streamlit App
st.title("🩺 Medical Chatbot")
st.write("Ask a medical question, and I'll try to provide a fact-based answer from Neo4j. If not available, I'll generate an answer using GPT-2.")

user_input = st.text_input("Enter your medical question:")

if st.button("Get Answer"):
    if user_input:
        response = chatbot_response(user_input)
        st.write(response)
    else:
        st.write("Please enter a question.")