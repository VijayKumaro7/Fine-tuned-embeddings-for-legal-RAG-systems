"""
Fine-tuning Module — Legal Embedding Model
Fine-tunes a sentence-transformer model on legal domain data
using MultipleNegativesRankingLoss for retrieval tasks.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_MODEL_PATH = "models/finetuned/legal-embedding-model"
BATCH_SIZE = 16
EPOCHS = 3
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 256

Path(OUTPUT_MODEL_PATH).mkdir(parents=True, exist_ok=True)


# ─── Training Pair Generation ───────────────────────────────────────────────────

def load_qa_pairs(path: str = "data/raw/legal_qa_pairs.json") -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def load_corpus(path: str = "data/raw/legal_corpus.json") -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def create_pairs_from_qa(qa_pairs: List[dict]) -> List[InputExample]:
    """
    Create (query, positive_document) training pairs from QA data.
    MultipleNegativesRankingLoss treats all other docs in the batch as negatives.
    """
    examples = []
    for pair in qa_pairs:
        q = pair.get("question", "").strip()
        a = pair.get("answer", "").strip()
        if q and a and len(a) > 50:
            examples.append(InputExample(texts=[q, a]))
    logger.info(f"Created {len(examples)} QA-based training pairs")
    return examples


def create_pairs_from_corpus(corpus: List[dict], n_pairs: int = 2000) -> List[InputExample]:
    """
    Create positive pairs from adjacent chunks of the same document.
    Adjacent sentences from the same legal document are semantically related.
    """
    examples = []
    texts = [doc["text"] for doc in corpus if len(doc["text"]) > 500]

    for text in random.sample(texts, min(n_pairs, len(texts))):
        # Split into sentences/chunks and pair adjacent ones
        sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 80]
        for i in range(len(sentences) - 1):
            examples.append(InputExample(texts=[sentences[i], sentences[i + 1]]))
            if len(examples) >= n_pairs:
                return examples

    logger.info(f"Created {len(examples)} corpus-based training pairs")
    return examples


def create_hard_negatives() -> List[InputExample]:
    """
    Manually crafted hard negative pairs for legal domain disambiguation.
    These teach the model domain-specific word meanings.
    """
    pairs = [
        # Legal "consideration" vs general usage
        ("What is consideration in a contract?",
         "Consideration is something of value exchanged between parties in a contract, such as money, services, or a promise to act."),
        # Legal "party" vs general
        ("Who are the parties in a lawsuit?",
         "In litigation, the party refers to a person or entity involved in the legal proceedings, either as plaintiff or defendant."),
        # Legal "execution" of a contract
        ("What does executing a contract mean?",
         "Executing a contract means signing and completing all formalities required to make it legally binding and enforceable."),
        # Legal "discovery"
        ("What is discovery in civil litigation?",
         "Discovery is the pre-trial process where parties exchange relevant information, documents, and evidence before a civil trial."),
        # Legal "brief"
        ("What is a legal brief?",
         "A legal brief is a written argument submitted to a court that presents facts, legal arguments, and citations to support a party's position."),
        # Legal "standing"
        ("What is standing in constitutional law?",
         "Standing requires a plaintiff to demonstrate a concrete injury, causation by the defendant's conduct, and redressability by a court ruling."),
        # Legal "relief"
        ("What kinds of relief can a court grant?",
         "Courts may grant equitable relief such as injunctions, declaratory relief, or monetary damages including compensatory and punitive damages."),
    ]

    examples = []
    for q, a in pairs:
        examples.append(InputExample(texts=[q, a]))

    logger.info(f"Created {len(examples)} hard negative pairs")
    return examples


def prepare_training_data() -> List[InputExample]:
    """Combine all training sources."""
    all_examples = []

    try:
        qa_pairs = load_qa_pairs()
        all_examples += create_pairs_from_qa(qa_pairs)
    except FileNotFoundError:
        logger.warning("QA pairs file not found. Skipping.")

    try:
        corpus = load_corpus()
        all_examples += create_pairs_from_corpus(corpus, n_pairs=1500)
    except FileNotFoundError:
        logger.warning("Corpus file not found. Skipping.")

    all_examples += create_hard_negatives()
    random.shuffle(all_examples)
    logger.info(f"Total training examples: {len(all_examples)}")
    return all_examples


# ─── Evaluation Setup ───────────────────────────────────────────────────────────

def build_evaluator() -> InformationRetrievalEvaluator:
    """Build an IR evaluator with legal domain queries."""
    queries = {
        "q1": "What are the elements of a valid contract?",
        "q2": "When can a court grant an injunction?",
        "q3": "Define mens rea in criminal law",
        "q4": "What is breach of fiduciary duty?",
        "q5": "How is damages calculated in contract breach?",
    }

    corpus_eval = {
        "d1": "A valid contract requires offer, acceptance, consideration, capacity, and legality. Each element must be present for enforcement.",
        "d2": "An injunction is equitable relief granted when monetary damages are inadequate, the plaintiff shows irreparable harm, and the balance of hardships favors relief.",
        "d3": "Mens rea, or guilty mind, refers to the criminal intent required for most crimes. It distinguishes intentional wrongdoing from accidents.",
        "d4": "Breach of fiduciary duty occurs when a person in a position of trust fails to act in the best interest of the beneficiary, causing harm.",
        "d5": "Damages for breach of contract aim to put the non-breaching party in the position they would have been in had the contract been performed.",
        "d6": "Promissory estoppel prevents a party from withdrawing a promise when the other party reasonably relied on it to their detriment.",
        "d7": "The statute of frauds requires certain contracts to be in writing, including those for real estate, marriage, goods over $500, and agreements lasting over one year.",
    }

    relevant_docs = {
        "q1": {"d1"},
        "q2": {"d2"},
        "q3": {"d3"},
        "q4": {"d4"},
        "q5": {"d5"},
    }

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus_eval,
        relevant_docs=relevant_docs,
        name="legal-ir-eval",
    )


# ─── Fine-tuning ────────────────────────────────────────────────────────────────

def finetune_embedding_model():
    """Main fine-tuning function."""
    logger.info(f"Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH

    # Prepare data
    train_examples = prepare_training_data()

    if not train_examples:
        logger.error("No training examples found. Run data_collection.py first.")
        return

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # Loss: MultipleNegativesRankingLoss is best for retrieval tasks
    # It treats all other (query, doc) pairs in the batch as negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluator
    evaluator = build_evaluator()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")
    model.to(device)

    total_steps = len(train_dataloader) * EPOCHS

    logger.info(f"Starting fine-tuning: {EPOCHS} epochs, {total_steps} total steps")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        evaluation_steps=len(train_dataloader) // 2,  # eval twice per epoch
        output_path=OUTPUT_MODEL_PATH,
        show_progress_bar=True,
        save_best_model=True,
    )

    logger.info(f"Fine-tuning complete! Model saved to: {OUTPUT_MODEL_PATH}")
    return model


if __name__ == "__main__":
    finetune_embedding_model()
