from pathlib import Path
import gradio as gr
from gradio.themes.base import Base

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


def prepare_documents(folder):
    docs = []
    for file in Path(folder).glob('*.pdf'):
        docs.extend(PyPDFLoader(str(file)).load())   
    for file in Path(folder).glob('*.docx'):
        docs.extend(UnstructuredWordDocumentLoader(str(file)).load())
    docs.extend(DirectoryLoader(folder, loader_cls=TextLoader, glob="**/*.txt").load())
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)


def init_or_load_vectorstore(path, docs, embedding):
    if Path(path).exists():
        print("Loading vector store")
        return Chroma(collection_name="rag_chat", embedding_function=embedding, persist_directory=path)
    else:
        print("Creating new vector store")
        return Chroma.from_documents(documents=docs, embedding=embedding, collection_name="rag_chat", persist_directory=path)


def build_conversational_chain(llm, retriever, memory):
    prompt = PromptTemplate.from_template(
        """
        You are an assistant for technical troubleshooting and product support.
        Use ONLY the provided context retrieved from documents relevant to the user's specific question.
        - Always tell the truth.
        - If you don't know the answer, say so clearly.
        - ALWAYS cite sources using the format [sourceX], where X is the number assigned to the source in this specific context. 
        Do NOT continue numbering from previous questions – always restart from [source1] for each new user question.
        - If the question is unclear or lacks detail, ask a follow-up question to improve your understanding.

        Chat History:
        {chat_history}

        Question:
        {question}

        Context:
        {context}

        Answer:
        """
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain


if __name__ == "__main__":
    print("Launching session chat with memory")
    model_name = "llama3" #Chose the LLM model you want to use.
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2") #Chose what Embedding model you want to use.
    base_dir = Path(__file__).resolve().parent.parent
    docs_path = str(base_dir / "docs")
    db_path = str(base_dir / "chroma_db")

    vectorstore = init_or_load_vectorstore(db_path, prepare_documents(docs_path), embeddings)
    retriever = vectorstore.as_retriever()
    retriever.search_kwargs["k"] = 4  
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    qa = build_conversational_chain(ChatOllama(model=model_name, temperature=0.1), retriever, memory)

    def chat_with_memory(user_input, history):
        result = qa({"question": user_input})
        answer = result["answer"]
        sources = result.get("source_documents", [])

        formatted_sources = "\n\n".join(
            f"[source{idx+1}] {doc.metadata.get('source', 'unknown')}" for idx, doc in enumerate(sources)
        )

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        return history, history, formatted_sources, ""

    with gr.Blocks(theme=Base(), title="Session Chat with Memory") as app:
        gr.Markdown("# Conversational RAG Chat")
        chatbot = gr.Chatbot(type="messages")
        question_input = gr.Textbox(label="Ask a question:")
        reference_box = gr.Textbox(label="References", interactive=False)
        state = gr.State([])

        question_input.submit(chat_with_memory, [question_input, state], [chatbot, state, reference_box, question_input])

    app.launch(server_name="0.0.0.0", server_port=7860)