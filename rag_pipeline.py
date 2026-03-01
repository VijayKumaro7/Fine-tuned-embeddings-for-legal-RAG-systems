"""
Dual RAG Pipeline — Generic vs Fine-tuned Embeddings
Builds and runs two parallel RAG pipelines for comparison.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
GENERIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNED_MODEL = "models/finetuned/legal-embedding-model"
CORPUS_PATH = "data/raw/legal_corpus.json"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5


# ─── Document Loading & Chunking ───────────────────────────────────────────────

def load_corpus_as_langchain_docs(path: str = CORPUS_PATH) -> List[Document]:
    """Load saved legal corpus as LangChain Document objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for item in raw:
        docs.append(Document(
            page_content=item["text"],
            metadata={
                "source": item.get("source", "unknown"),
                **item.get("metadata", {})
            }
        ))
    logger.info(f"Loaded {len(docs)} documents from corpus")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ─── Vector Store Construction ─────────────────────────────────────────────────

def build_vectorstore(
    chunks: List[Document],
    embedding_model_name: str,
    collection_name: str,
    persist_dir: str = "data/processed/chroma",
) -> Chroma:
    """Build and persist a Chroma vector store with given embeddings."""
    logger.info(f"Building vectorstore: '{collection_name}' with model '{embedding_model_name}'")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    logger.info(f"Vectorstore '{collection_name}' built with {vectorstore._collection.count()} vectors")
    return vectorstore


# ─── RAG Chain Builder ──────────────────────────────────────────────────────────

LEGAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert legal assistant. Use the following legal documents to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Provide a precise legal answer based on the context. If the context doesn't contain enough information, say so.

Answer:"""
)


def build_rag_chain(vectorstore: Chroma, llm) -> RetrievalQA:
    """Build a RetrievalQA chain from a vectorstore."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": LEGAL_PROMPT},
    )
    return chain


# ─── Pipeline Manager ──────────────────────────────────────────────────────────

class DualRAGPipeline:
    """
    Manages two RAG pipelines — generic and fine-tuned — for side-by-side comparison.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.generic_vectorstore: Optional[Chroma] = None
        self.finetuned_vectorstore: Optional[Chroma] = None
        self.generic_chain: Optional[RetrievalQA] = None
        self.finetuned_chain: Optional[RetrievalQA] = None

    def build(self, corpus_path: str = CORPUS_PATH):
        """Build both pipelines end-to-end."""
        docs = load_corpus_as_langchain_docs(corpus_path)
        chunks = chunk_documents(docs)

        # Build vectorstores
        self.generic_vectorstore = build_vectorstore(
            chunks, GENERIC_MODEL, "legal_generic"
        )
        self.finetuned_vectorstore = build_vectorstore(
            chunks, FINETUNED_MODEL, "legal_finetuned"
        )

        # Build chains (only if LLM provided)
        if self.llm:
            self.generic_chain = build_rag_chain(self.generic_vectorstore, self.llm)
            self.finetuned_chain = build_rag_chain(self.finetuned_vectorstore, self.llm)

        logger.info("Both RAG pipelines built successfully!")

    def retrieve_comparison(self, query: str, k: int = 5) -> Dict:
        """
        Compare retrieved documents from both pipelines for a given query.
        Shows which documents each embedding model retrieves.
        """
        generic_docs = self.generic_vectorstore.similarity_search_with_score(query, k=k)
        finetuned_docs = self.finetuned_vectorstore.similarity_search_with_score(query, k=k)

        return {
            "query": query,
            "generic": [
                {"content": doc.page_content[:300], "score": float(score), "source": doc.metadata.get("source")}
                for doc, score in generic_docs
            ],
            "finetuned": [
                {"content": doc.page_content[:300], "score": float(score), "source": doc.metadata.get("source")}
                for doc, score in finetuned_docs
            ],
        }

    def answer_comparison(self, query: str) -> Dict:
        """Run both chains and return answers side by side."""
        if not (self.generic_chain and self.finetuned_chain):
            raise ValueError("LLM not configured. Initialize with llm= parameter.")

        generic_result = self.generic_chain({"query": query})
        finetuned_result = self.finetuned_chain({"query": query})

        return {
            "query": query,
            "generic_answer": generic_result["result"],
            "finetuned_answer": finetuned_result["result"],
            "generic_sources": [d.metadata.get("source") for d in generic_result["source_documents"]],
            "finetuned_sources": [d.metadata.get("source") for d in finetuned_result["source_documents"]],
        }


# ─── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo: show retrieval differences without an LLM
    pipeline = DualRAGPipeline()
    pipeline.build()

    test_queries = [
        "What is consideration in contract formation?",
        "When can a court award punitive damages?",
        "Define breach of fiduciary duty",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        result = pipeline.retrieve_comparison(query)

        print("\n--- Generic Model Retrieved ---")
        for i, doc in enumerate(result["generic"], 1):
            print(f"{i}. [score={doc['score']:.4f}] {doc['content'][:150]}...")

        print("\n--- Fine-tuned Model Retrieved ---")
        for i, doc in enumerate(result["finetuned"], 1):
            print(f"{i}. [score={doc['score']:.4f}] {doc['content'][:150]}...")
