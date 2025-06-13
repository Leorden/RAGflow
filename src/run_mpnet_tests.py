import os
import time
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# ---- INSTÄLLNINGAR ----
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
llms_to_test = ["mistral:instruct", "llama3", "openhermes", "zephyr"]
docs_path = "../docs"
db_path = "../chroma_db"
output_file = "testresult_all-MiniLM-L6-v2.xlsx"

questions = [
    "What was the root cause of the transformer failure in the SmartKraft DC units reported in Romania?",
    "How does the SmartKraft 2 system function according to the block diagram, and what are the main energy conversion stages?",
    "What is the function of spark detection and voltage reduction in SmartKraft operation?",
    "What temperatures were recorded for IGBT components during the heat tests in reports REPO-2942 and REPO-2944, and how stable were they?",
    "What maintenance actions are recommended to ensure optimal performance and avoid hardware failures in SmartKraft units?",
    "How is the SmartKraft 2 control system structured regarding I/O signals, control interface, and communication?",
    "What torque values and installation instructions apply to final mounting of HF coils and feedthrough plates as per INST-1433?",
    "What is the impact of the air filter and oil cooler on system cooling and performance, and how should they be maintained?"
]

def prepare_documents(folder):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(os.path.join(folder, file)).load())
        elif file.endswith(".docx"):
            docs.extend(UnstructuredWordDocumentLoader(os.path.join(folder, file)).load())
    docs.extend(DirectoryLoader(folder, loader_cls=TextLoader, glob="**/*.txt").load())
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

def build_chain(llm, retriever):
    prompt = PromptTemplate.from_template("""
    You are an assistant for technical troubleshooting and product support.
    Use the following context retrieved from documents to answer the user's question.
    - Always tell the truth.
    - If you don't know the answer, say so clearly.
    - Always cite sources using the format [sourceX], where X matches the listed document number.
    - If the question is unclear or lacks detail, ask a follow-up question to improve your understanding.

    Chat History:
    {chat_history}

    Question:
    {question}

    Context:
    {context}

    Answer:
    """)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# Skapa vektorbasen en gång om den inte redan finns
if not os.path.exists(db_path):
    print("Creating vectorstore for embedding model...")
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    chunks = prepare_documents(docs_path)
    Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=db_path, collection_name="rag_chat")
else:
    print("Using existing ChromaDB.")

# Ladda retriever (delad mellan alla LLMs)
retriever = Chroma(
    collection_name="rag_chat",
    embedding_function=HuggingFaceEmbeddings(model_name=embedding_model_name),
    persist_directory=db_path
).as_retriever()

# --- HUVUDLOOP ---
results = []

for llm_name in llms_to_test:
    print(f"\n== Testing LLM: {llm_name} ==")
    llm = ChatOllama(model=llm_name, temperature=0.1)
    chain = build_chain(llm, retriever)

    for q in questions:
        start_time = time.time()
        response = chain({"question": q})
        end_time = time.time()
        elapsed = round(end_time - start_time, 2)
        results.append({
            "Embeddingmodell": embedding_model_name,
            "LLM": llm_name,
            "Fråga": q,
            "Modellens Svar": response["answer"],
            "Svarstid (s)": elapsed
        })

# Spara resultat till Excel
df = pd.DataFrame(results)
df.to_excel(output_file, index=False)
print(f"\nAll tests completed. Results saved to: {output_file}")