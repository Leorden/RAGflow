import os
import glob

import gradio as gr
from gradio.themes.base import Base

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def prepare_documents(input_docs_dir):
    docs = []

    # PDF
    for file in glob.glob(os.path.join(input_docs_dir, "*.pdf")):
        loader = PyPDFLoader(file)
        docs.extend(loader.load())

    # DOCX
    for file in glob.glob(os.path.join(input_docs_dir, "*.docx")):
        loader = UnstructuredWordDocumentLoader(file)
        docs.extend(loader.load())

    # TXT
    loader = DirectoryLoader(input_docs_dir, loader_cls=TextLoader, glob="**/*.txt")
    docs.extend(loader.load())

    # split texts into chunks with overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    return splits


def initialize_vectorstore(db_path, docs, embed_model):
    print("Creating new vector index...")
    return Chroma.from_documents(
        documents=docs,
        collection_name="rag_data",
        embedding=embed_model,
        persist_directory=db_path
    ).as_retriever()


def load_vectorstore(db_path, embed_model):
    print("Loading existing vector index...")
    return Chroma(
        collection_name="rag_data",
        embedding_function=embed_model,
        persist_directory=db_path
    ).as_retriever()


def fetch_relevant_docs(retriever, user_query):
    print("Running retrieval...")
    return retriever.invoke(user_query)


def assemble_context(docs):
    content = ""
    for index, document in enumerate(docs):
        content = content + "[doc" + str(index + 1) + "]=" + document.page_content.replace("\n", " ") + "\n\n"
    return content


def assemble_sources(docs):
    sources = ""
    for index, document in enumerate(docs):
        sources = sources + "[doc" + str(index + 1) + "]=" + document.metadata.get("source", "unknown") + "\n\n"
    return sources.strip()


def generate_answer(query, context, llm_name):
    print("Generating response...")
    prompt_template = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Keep the answer concise, truthful, and informative. 
If you decide to use a source, you must mention in which document you found specific information. 
Sources are indicated in the context by [doc<doc_number>].

Question: {question} 
Context: {context} 
Answer:"""
    )
    llm = ChatOllama(model=llm_name, temperature=0)
    rag_chain = prompt_template | llm | StrOutputParser()
    return rag_chain.invoke({"context": context, "question": query})


if __name__ == "__main__":
    print("Launching app...")

    llm_model = "mistral:instruct"
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    input_docs_dir = '../docs'
    vector_db_dir = '../chroma_db'

    if not os.path.exists(vector_db_dir):
        retriever = initialize_vectorstore(vector_db_dir, prepare_documents(input_docs_dir), embedder)
    else:
        retriever = load_vectorstore(vector_db_dir, embedder)

    def rag_pipeline(user_input):
        docs = fetch_relevant_docs(retriever, user_input)
        context = assemble_context(docs)
        response = generate_answer(user_input, context, llm_model)
        sources = assemble_sources(docs)
        return sources, response

    with gr.Blocks(theme=Base(), title="RAG Document Chat") as interface:
        gr.Markdown("# Chat with Your Documents")
        user_question = gr.Textbox(label="Enter your question:")
        with gr.Row():
            ask_button = gr.Button("Ask", variant="primary")
        with gr.Column():
            ref_output = gr.Textbox(lines=1, max_lines=10, label="References")
            answer_output = gr.Textbox(lines=1, max_lines=10, label="Response")

        ask_button.click(rag_pipeline, user_question, outputs=[ref_output, answer_output])

    interface.launch()
