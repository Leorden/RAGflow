import os
import glob

import gradio as gr
from gradio.themes.base import Base


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def chunking(data_directory):
    docs = []

    # PDF
    for file in glob.glob(os.path.join(data_directory, "*.pdf")):
        loader = PyPDFLoader(file)
        docs.extend(loader.load())

    # DOCX
    for file in glob.glob(os.path.join(data_directory, "*.docx")):
        loader = UnstructuredWordDocumentLoader(file)
        docs.extend(loader.load())

    # TXT
    loader = DirectoryLoader(data_directory, loader_cls=TextLoader, glob="**/*.txt")
    docs.extend(loader.load())

    # split texts into chunks with overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    return splits


def create_vector_store(db_directory, chunks, embedding):
    print("Creating vector store (this may take a while)")
    vectorstore = Chroma.from_documents(documents=chunks, collection_name="chromemwah", embedding=embedding,
                                        persist_directory=db_directory)
    return vectorstore.as_retriever()


def fetch_vector_store(db_directory, embedding):
    print("Fetching vector store")
    vectorstore = Chroma(collection_name="chromemwah", embedding_function=embedding, persist_directory=db_directory)
    return vectorstore.as_retriever()


def retrieve(retrieving, question):
    print("Retrieving")
    documents = retrieving.invoke(question)
    return documents


def context_formatting(documents):
    content = ""
    for index, document in enumerate(documents):
        content = content + "[doc" + str(index + 1) + "]=" + document.page_content.replace("\n", " ") + "\n\n"
    return content


def source_formatting(documents):
    sources = ""
    for index, document in enumerate(documents):
        sources = sources + "[doc" + str(index + 1) + "]=" + document.metadata.get("source", "unknown") + "\n\n"
    return sources.strip()


def generate(question, documents, use_llm):
    print("Generating")
    rag_prompt = ChatPromptTemplate.from_template("You are an assistant for question-answering tasks. Use the "
                                                  "following pieces of retrieved context to answer the question. If "
                                                  "you don't know the answer, just say that you don't know. Keep the "
                                                  "answer concise, truthful, and informative. If you decide to use a "
                                                  "source, you must mention in which document you found specific "
                                                  "information. Sources are indicated in the context by "
                                                  "[doc<doc_number>].\n"
                                                  "Question: {question} \n"
                                                  "Context: {context} \n"
                                                  "Answer:")
    llm = ChatOllama(model=use_llm, temperature=0)
    chain = rag_prompt | llm | StrOutputParser()
    output = chain.invoke({"context": documents, "question": question})
    return output


if __name__ == "__main__":
    print("Starting program")

    use_llm = "mistral:instruct"
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    data_directory = '../docs'
    db_directory = '../chroma_db'

    retriever = None
    if not os.path.isdir(db_directory):
        retriever = create_vector_store(db_directory, chunking(data_directory), embedding)
    else:
        retriever = fetch_vector_store(db_directory, embedding)

    def complete_rag(question):
        docs = retrieve(retriever, question)
        output = generate(question, context_formatting(docs), use_llm)
        return source_formatting(docs), output

    with gr.Blocks(theme=Base(), title="Q&A on your data with RAG") as demo:
        gr.Markdown("# Q&A on your data with RAG")
        textbox = gr.Textbox(label="Question:")
        with gr.Row():
            button = gr.Button("Submit", variant="primary")
        with gr.Column():
            output1 = gr.Textbox(lines=1, max_lines=10, label="Sources")
            output2 = gr.Textbox(lines=1, max_lines=10, label="Output")

        button.click(complete_rag, textbox, outputs=[output1, output2])

    demo.launch()
