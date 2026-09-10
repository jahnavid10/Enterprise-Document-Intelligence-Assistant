import os
import faiss
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class RAGPipeline:
    def __init__(
        self,
        index_path="data/bbc_faiss_index.index",
        metadata_path="data/bbc_chunk_metadata.csv",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name="google/flan-t5-small",
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name

        self.index = self.load_faiss_index()
        self.metadata = self.load_metadata()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)
        self.llm_model = self.llm_model.to(self.device)
        self.llm_model.eval()

    def load_faiss_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        return faiss.read_index(self.index_path)

    def load_metadata(self):
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        metadata = pd.read_csv(self.metadata_path)

        if len(metadata) != self.index.ntotal:
            raise ValueError("Metadata rows do not match FAISS index size.")

        return metadata

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for rank, idx in enumerate(indices[0], start=1):
            row = self.metadata.iloc[idx]

            results.append({
                "rank": rank,
                "score": float(scores[0][rank - 1]),
                "title": str(row["title"]),
                "chunk_text": str(row["chunk_text"]),
            })

        return pd.DataFrame(results)

    def build_prompt(self, question, retrieved_chunks):
        context = ""

        for _, row in retrieved_chunks.iterrows():
            context += f"Title: {row['title']}\n"
            context += f"News text: {row['chunk_text']}\n\n"

        prompt = f"""
You are a BBC news research assistant.

Using only the retrieved BBC news articles, write a detailed answer to the user's question.

Structure your answer like this:

1. Start with a short direct answer.
2. Then explain the main news updates in 2 short paragraphs.
3. Mention important article titles or events where relevant.
4. End with a short overall summary.

Do not show technical details such as rank, score, chunk ID, FAISS, embeddings, or category.
Do not say "unknown".

Retrieved BBC News Articles:
{context}

User Question:
{question}

Answer:
"""
        return prompt

    def generate_answer(self, prompt, max_new_tokens=250):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def fallback_summary(self, retrieved_chunks):
        """
        Creates a readable extractive answer if the LLM gives a weak response.
        """
        top_articles = retrieved_chunks.head(3)

        summary_lines = []

        for _, row in top_articles.iterrows():
            text = row["chunk_text"]
            sentences = text.split(". ")

            useful_sentences = sentences[:2]

            for sentence in useful_sentences:
                sentence = sentence.strip()
                if len(sentence) > 40:
                    summary_lines.append(sentence)

        if not summary_lines:
            return "The retrieved news articles discuss relevant updates, but there is not enough clear text to generate a detailed summary."

        answer = " ".join(summary_lines[:6])

        if not answer.endswith("."):
            answer += "."

        return answer

    def clean_sources(self, retrieved_chunks):
        """
        Returns only article titles, without rank, score, category, or chunk ID.
        """
        titles = []

        for title in retrieved_chunks["title"].tolist():
            title = str(title).strip()

            if title and title.lower() not in ["nan", "unknown", "untitled article"]:
                titles.append(title)

        unique_titles = list(dict.fromkeys(titles))

        if not unique_titles:
            return "No article titles available."

        return "\n".join([f"- {title}" for title in unique_titles[:5]])

    def answer_question(self, question, top_k=5):
        retrieved_chunks = self.retrieve(question, top_k=top_k)

        prompt = self.build_prompt(question, retrieved_chunks)
        answer = self.generate_answer(prompt)

        weak_answers = [
            "i do not have enough information",
            "not enough information",
            "i don't know",
            "unknown",
        ]

        if any(text in answer.lower() for text in weak_answers) or len(answer.split()) < 20:
            answer = self.fallback_summary(retrieved_chunks)

        sources = self.clean_sources(retrieved_chunks)

        return answer, sources