"""
Data Collection Module — Legal Domain RAG
Collects legal documents from HuggingFace datasets and local sources.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from datasets import load_dataset
from langchain.schema import Document
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_pile_of_law(split: str = "train", max_samples: int = 5000) -> List[Dict]:
    """
    Load legal documents from the Pile of Law dataset on HuggingFace.
    Covers: court opinions, contracts, statutes, terms of service.
    """
    logger.info(f"Loading Pile of Law dataset (max {max_samples} samples)...")

    docs = []
    subsets = ["tos", "courtlistener_opinions", "atticus_contracts"]

    for subset in subsets:
        try:
            dataset = load_dataset(
                "pile-of-law/pile-of-law",
                subset,
                split=split,
                streaming=True,
                trust_remote_code=True
            )
            count = 0
            for item in dataset:
                if count >= max_samples // len(subsets):
                    break
                text = item.get("text", "").strip()
                if len(text) > 200:  # filter out very short docs
                    docs.append({
                        "text": text,
                        "source": subset,
                        "metadata": {
                            "dataset": "pile-of-law",
                            "subset": subset,
                        }
                    })
                    count += 1
            logger.info(f"  Loaded {count} docs from '{subset}'")
        except Exception as e:
            logger.warning(f"  Could not load subset '{subset}': {e}")

    logger.info(f"Total documents collected: {len(docs)}")
    return docs


def load_legal_qa_dataset(max_samples: int = 2000) -> List[Dict]:
    """
    Load legal Q&A pairs for generating fine-tuning training pairs.
    Returns question-answer pairs from legal domains.
    """
    logger.info("Loading Legal QA dataset...")
    pairs = []

    try:
        dataset = load_dataset("nguyen-brat/legal_qa", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            if question and answer and len(answer) > 100:
                pairs.append({"question": question, "answer": answer})
        logger.info(f"Loaded {len(pairs)} QA pairs")
    except Exception as e:
        logger.warning(f"Could not load legal QA dataset: {e}. Using synthetic pairs.")
        pairs = generate_synthetic_pairs()

    return pairs


def generate_synthetic_pairs() -> List[Dict]:
    """Fallback: generate synthetic legal QA pairs for training."""
    return [
        {
            "question": "What constitutes a valid contract under common law?",
            "answer": "A valid contract requires offer, acceptance, consideration, capacity, and legality. "
                      "The offer must be clear and definite. Acceptance must mirror the offer exactly. "
                      "Consideration means each party must exchange something of value.",
        },
        {
            "question": "What is the difference between a void and voidable contract?",
            "answer": "A void contract has no legal effect and cannot be enforced by either party. "
                      "A voidable contract is initially valid but can be rescinded by one party, "
                      "typically due to fraud, duress, undue influence, or misrepresentation.",
        },
        {
            "question": "Define promissory estoppel in contract law.",
            "answer": "Promissory estoppel prevents a promisor from denying a promise when the promisee "
                      "reasonably relied on it to their detriment. The doctrine applies when there is a "
                      "clear promise, reasonable reliance, and injustice can only be avoided by enforcement.",
        },
        {
            "question": "What are the elements of negligence in tort law?",
            "answer": "To establish negligence, a plaintiff must prove: duty of care owed by the defendant, "
                      "breach of that duty, causation (both actual and proximate), and damages suffered. "
                      "The standard of care is usually that of a reasonable person in similar circumstances.",
        },
        {
            "question": "When can a court pierce the corporate veil?",
            "answer": "Courts may pierce the corporate veil to hold shareholders personally liable when the "
                      "corporation is used as an alter ego, there is commingling of assets, undercapitalization, "
                      "failure to follow corporate formalities, or when fraud is involved.",
        },
    ]


def save_documents(docs: List[Dict], filename: str) -> None:
    """Save collected documents to JSON."""
    output_path = RAW_DATA_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(docs)} documents to {output_path}")


def load_saved_documents(filename: str) -> List[Dict]:
    """Load previously saved documents."""
    path = RAW_DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # Collect corpus documents
    corpus = load_pile_of_law(max_samples=3000)
    save_documents(corpus, "legal_corpus.json")

    # Collect QA pairs for fine-tuning
    qa_pairs = load_legal_qa_dataset(max_samples=2000)
    save_documents(qa_pairs, "legal_qa_pairs.json")

    logger.info("Data collection complete!")
