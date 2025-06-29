png
# 🔎 RAGflow – Industrial Document Search

<img src="docs/Local_RAG.png" width="700" alt="System architecture" />

A local Retrieval-Augmented Generation (RAG) prototype that indexes internal documents, retrieves the most relevant chunks via **ChromaDB**, and lets a locally-hosted LLM answer user questions.  
Built during my bachelor thesis at University West in collaboration with KraftPowercon.

---

## ✨ Features
| Step | Description |
|------|-------------|
| **1** | Load documents from `docs/`, split into chunks |
| **2** | Embed all chunks with a Hugging-Face embedding model |
| **3** | Vectorize the user query with the *same* embedding model |
| **4-5** | Query **ChromaDB** to get top-k similar chunks |
| **6** | Combine query + chunks → prompt a local LLM via **Ollama** |
| **7** | Stream the answer back to the user (citations included) |

---

## 🧱 Repository structure

```text
RAGflow/
├─ chroma_db/                # auto-generated vector database (ignored)
├─ docs/                     # your manual files & architecture diagram
│   └─ workflow.png
├─ src/                      # source code
│   ├─ rag_chat_session.py   # main chat app (with history)
│   ├─ rag_chat_first.py     # initial POC chat (no memory)
│   └─ run_mpnet_tests.py    # benchmark script
├─ requirements.txt          # Python dependencies
└─ README.md                 # this file
```
---

## 🔧 Requirements

* Python 3.9+  
* **Ollama** for local LLM hosting → <https://ollama.ai>  
* A Sentence-Transformer embedding model (downloaded the first time you run)

---

## ⚙️ Installation & Usage
1. Clone and install dependencies
```bash
git clone https://github.com/Leorden/RAGflow.git
cd RAGflow
pip install -r requirements.txt
```
2. Start a local LLM with Ollama
```bash
ollama run <model_name>:<tag>
```
Example:
```bash
ollama run mistral:instruct
```
3. Download embedding model and configure script
```bash
from sentence_transformers import SentenceTransformer
SentenceTransformer("intfloat/e5-base-v2")
```
4. Edit model names in src/rag_chat_session.py
```pyhton
model_name = "mistral:instruct"
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
```
5. Add your documents
- Place the documents you want the RAG system to use in the `/docs` folder.  
- Supported formats: `.pdf`, `.docx`, `.txt`

6. Run the chat application
```bash
python src/rag_chat_session.py
```

- First run builds the vector store (chroma_db/, may take a while)
- Next runs are fast using cached embeddings

## 🧪 Benchmark script
To evaluate the response quality of different LLMs using one embedding model, use:
```bash
python src/run_mpnet_tests.py
```
This script:

- Embeds all documents using one embedding model (e.g. all-MiniLM-L6-v2)

- Loops through a list of LLMs (you configure them)

- Asks a list of questions to each LLM

- Outputs results to an .xlsx file in /src/

## 🛠 How to configure the benchmark
Open src/run_mpnet_tests.py and edit the following:
```python
# Embedding model to use
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# LLMs to test via Ollama
llms_to_test = ["mistral:instruct", "llama3", "openhermes", "zephyr"]

# Questions to ask
questions = [
    "How do I make a rainbow?",
    "What currency do they have in Canada?"
]
```
⚠️ Only one embedding model can be tested at a time.

The script will generate a file like: testresult_all-MiniLM-L6-v2.xlsx

in the /src/ folder (ignored from Git). This file contains a row for each LLM and question pair, with the generated response and time generate the response.

## 📝 Notes
 rag_chat_first.py is an early proof-of-concept (single-turn chat).

 Use rag_chat_session.py for memory across turns.

All large folders (chroma_db/, LLMs/, Embeddingmodeller/, *.xlsx) are git-ignored to keep the repo clean.

## 🗺 Roadmap
 Add Dockerfile for one-command setup

 Introduce streaming responses via WebSockets

 Optional support for OpenAI / remote LLMs

## 📜 License
This project is released under the MIT License – see LICENSE.

## 🧑‍🤝‍🧑 Contributors

| Role | Name |
|------|------|
| Maintainer | [Johan Froissart](https://www.linkedin.com/in/johan-froissart-6374b8139/) |
| Former contributor (2025-03 – 2025-06) | [Hubert Modh](https://www.linkedin.com/in/hubert-modh-2925b8265/) |

