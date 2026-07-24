# Enterprise Document Intelligence Assistant using LLM and RAG

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot built using BBC News articles.
The system retrieves relevant news article chunks using **FAISS semantic search** and generates grounded answers using a **Hugging Face language model**.

---

## Project Overview

This project demonstrates how Large Language Models can be combined with a retrieval system to answer questions from a document collection.

Instead of relying only on the LLM’s internal knowledge, the chatbot first retrieves relevant BBC News article chunks and then uses them as context to generate answers.

This helps improve answer relevance, provide document-grounded responses, and reduce hallucinations.

---

## Features

* End-to-end RAG pipeline
* BBC News document collection
* Text preprocessing
* Document chunking
* SentenceTransformer embeddings
* FAISS vector database
* Semantic search
* LLM-based answer generation
* Gradio chatbot interface
* Retrieved article titles shown with answers

---

## Tech Stack

* Python
* pandas
* NumPy
* FAISS
* SentenceTransformers
* Hugging Face Transformers
* PyTorch
* Gradio
* Kaggle Notebooks

---

## Project Pipeline

```text
BBC News Dataset
        ↓
Problem Framing and Data Understanding
        ↓
Text Preprocessing
        ↓
Document Chunking
        ↓
Embedding Generation
        ↓
FAISS Vector Indexing
        ↓
Semantic Retrieval
        ↓
RAG Answer Generation
        ↓
Gradio Chatbot
```

---

## Notebook Overview

The project was developed step by step using separate Kaggle notebooks. Each notebook focuses on one stage of the RAG pipeline and produces outputs that are used in the next stage.

---

### 01. Problem Framing and Data Understanding

**Notebook:** `01_Problem_Framing_Data_Understanding.ipynb`

This notebook introduces the project idea and prepares the raw BBC News dataset for the RAG pipeline.

**What this notebook covers:**

* Defined the project objective.
* Downloaded the BBC News dataset from Kaggle.
* Loaded the dataset using pandas.
* Explored dataset shape, columns, missing values, and duplicate rows.
* Standardized the dataset into a common document format.
* Created document-level fields such as `doc_id`, `title`, `category`, and `text`.
* Removed empty and duplicate documents.
* Added basic word count statistics.

**Output file:**

```text
bbc_docs_standard.csv
```

---

### 02. Text Preprocessing

**Notebook:** `02_Text_Preprocessing.ipynb`

This notebook cleans the article text while keeping the natural sentence structure useful for embedding models.

**What this notebook covers:**

* Loaded the standardized dataset from Notebook 1.
* Removed HTML tags.
* Removed URLs.
* Removed unnecessary line breaks and extra spaces.
* Normalized punctuation spacing.
* Preserved stopwords and natural sentence structure.
* Created a cleaned text column called `text_clean`.
* Removed very short articles.
* Compared original and cleaned text.

**Output file:**

```text
bbc_docs_preprocessed.csv
```

---

### 03. Document Chunking

**Notebook:** `03_Document_Chunking.ipynb`

This notebook splits long BBC articles into smaller overlapping chunks for better semantic retrieval.

**What this notebook covers:**

* Loaded the preprocessed dataset from Notebook 2.
* Applied word-based chunking.
* Used a chunk size of 180 words.
* Used an overlap of 40 words.
* Created chunk-level metadata.
* Generated fields such as `chunk_id`, `doc_id`, `chunk_index`, `title`, `category`, `chunk_text`, and `chunk_word_count`.
* Analyzed the number of chunks and chunk length distribution.

**Output file:**

```text
bbc_docs_chunks.csv
```

---

### 04. Embedding Generation

**Notebook:** `04_Embedding_Generation.ipynb`

This notebook converts each document chunk into a dense vector embedding.

**What this notebook covers:**

* Loaded the chunked dataset from Notebook 3.
* Used the `sentence-transformers/all-MiniLM-L6-v2` embedding model.
* Generated embeddings for every document chunk.
* Normalized embeddings for cosine similarity search.
* Saved embeddings separately from metadata.
* Verified that each chunk has a matching embedding.

**Output files:**

```text
bbc_chunk_embeddings.npy
bbc_chunk_metadata.csv
```

---

### 05. FAISS Vector Indexing

**Notebook:** `05_FAISS_Vector_Indexing.ipynb`

This notebook builds the FAISS vector database used for fast semantic search.

**What this notebook covers:**

* Loaded chunk embeddings and metadata.
* Created a FAISS index using `IndexFlatIP`.
* Added normalized embeddings to the FAISS index.
* Tested semantic search with a sample query.
* Verified that the FAISS index and metadata rows match.
* Saved the final FAISS index.

**Output file:**

```text
bbc_faiss_index.index
```

---

### 06. Retrieval System

**Notebook:** `06_Retrieval_System.ipynb`

This notebook builds the reusable retriever component of the RAG system.

**What this notebook covers:**

* Loaded the FAISS index.
* Loaded chunk metadata.
* Loaded the same SentenceTransformer model used during embedding generation.
* Converted user questions into query embeddings.
* Retrieved the top-k most relevant chunks from FAISS.
* Displayed retrieved article titles, categories, similarity scores, and text previews.
* Tested retrieval using multiple sample questions.

**Output file:**

```text
sample_retrieval_results.csv
```

---

### 07. RAG Pipeline

**Notebook:** `07_RAG_Pipeline.ipynb`

This notebook connects the retriever with a Large Language Model to create the complete RAG pipeline.

**What this notebook covers:**

* Loaded the FAISS index and chunk metadata.
* Retrieved relevant BBC News chunks for a user question.
* Built prompts using the retrieved context.
* Used `google/flan-t5-base` for answer generation.
* Generated grounded answers from retrieved BBC article chunks.
* Displayed the generated answer along with retrieved news article titles.
* Saved sample RAG outputs.

**Output file:**

```text
sample_rag_results.csv
```

---

### 08. RAG Evaluation

**Notebook:** `08_RAG_Evaluation.ipynb`

This notebook evaluates the retrieval and RAG system using simple, explainable metrics.

**What this notebook covers:**

* Loaded the FAISS index and chunk metadata.
* Created sample evaluation queries for different news topics.
* Measured retrieval latency.
* Checked Top-1, Top-3, and Top-5 retrieval quality.
* Reviewed generated RAG answers.
* Saved evaluation results.

**Output files:**

```text
retrieval_evaluation_results.csv
rag_answer_review.csv
```

---

## Final Chatbot Application

After completing the notebooks, the retrieval assets were moved into a Python application.

The final chatbot uses:

* `app.py` for the Gradio interface
* `rag_pipeline.py` for retrieval and answer generation
* `bbc_faiss_index.index` as the FAISS vector database
* `bbc_chunk_metadata.csv` as the chunk metadata file

---

## Models Used

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model converts document chunks and user queries into dense vector embeddings.

### Language Model

```text
google/flan-t5-base
```

This model generates answers using the retrieved document context.

---

## Repository Structure

```text
enterprise-document-intelligence-rag/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── bbc_faiss_index.index
│   └── bbc_chunk_metadata.csv
│
└── notebooks/
    ├── 01_Problem_Framing_Data_Understanding.ipynb
    ├── 02_Text_Preprocessing.ipynb
    ├── 03_Document_Chunking.ipynb
    ├── 04_Embedding_Generation.ipynb
    ├── 05_FAISS_Vector_Indexing.ipynb
    ├── 06_Retrieval_System.ipynb
    ├── 07_RAG_Pipeline.ipynb
    └── 08_RAG_Evaluation.ipynb
```

---

## How the Chatbot Works

1. The user asks a question.
2. The question is converted into an embedding using SentenceTransformers.
3. FAISS searches for the most relevant BBC article chunks.
4. Retrieved chunks are added to the prompt as context.
5. The LLM generates an answer based on the retrieved context.
6. The chatbot displays the answer and related article titles.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jahnavid10/enterprise-document-intelligence-rag.git
cd enterprise-document-intelligence-rag
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Chatbot Locally

```bash
python app.py
```

Open the Gradio link shown in the terminal:

```text
http://127.0.0.1:7860
```

---

## Requirements

```text
gradio
pandas
numpy
faiss-cpu
sentence-transformers
transformers
torch
accelerate
```

---

## Deployment

The chatbot can be deployed on **Hugging Face Spaces** using Gradio.

🔗 **Live Demo:** [Enterprise Document Intelligence Assistant](https://huggingface.co/spaces/dulalajahnavi10/enterprise-document-intelligence-rag)

Required files for deployment:
```text
app.py
rag_pipeline.py
requirements.txt
README.md
data/bbc_faiss_index.index
data/bbc_chunk_metadata.csv
```

---

## Evaluation

The retrieval system was evaluated using simple retrieval checks and latency measurements.

Evaluation included:

* Top-k retrieval quality
* Similarity score inspection
* Retrieval latency
* Manual answer review
* Source relevance checking

---

## Limitations

* The chatbot answers only from the stored BBC News dataset.
* It does not provide live or real-time news.
* Answer quality depends on the retrieved chunks.
* `google/flan-t5-base` is lightweight, so answers may be shorter than larger LLM outputs.
* The dataset may not contain all topics or recent events.

---

## Future Improvements

* Use a stronger LLM for richer responses.
* Add hybrid search using BM25 and FAISS.
* Add reranking for better retrieval quality.
* Add source URLs for article citation.
* Allow users to upload custom documents.
* Add live document ingestion.
* Improve RAG evaluation with groundedness and faithfulness metrics.

---

## Author

Developed by **Jahnavi Dulala**
