"""Central configuration. Every value can be overridden with an environment variable,
and the defaults match docker-compose.yml, so nothing needs to be set to run the project."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", REPO_ROOT))
DATA_DIR = Path(os.environ.get("DATA_DIR", WORKSPACE / "data"))
RAW_DIR = Path(os.environ.get("RAW_DIR", DATA_DIR / "raw"))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", DATA_DIR / "processed"))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", WORKSPACE / "models" / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", WORKSPACE / "results"))

POSTGRES = dict(
    host=os.environ.get("POSTGRES_HOST", "postgres"),
    port=int(os.environ.get("POSTGRES_PORT", "5432")),
    dbname=os.environ.get("POSTGRES_DB", "drug_safety_db"),
    user=os.environ.get("POSTGRES_USER", "drug_user"),
    password=os.environ.get("POSTGRES_PASSWORD", "drug_password"),
)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_AUTH = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "drugsafety123"))

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", str(24 * 3600)))

CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]

# Hugging Face model ids tried, in order, for each backbone (the first one found in the local cache is used)
BACKBONE_CANDIDATES = {
    "bert_base": ("BERT-base", ["bert-base-uncased", "google-bert/bert-base-uncased", "bert-base-cased"]),
    "roberta": ("RoBERTa", ["roberta-base", "FacebookAI/roberta-base"]),
    "scibert": ("SciBERT", ["allenai/scibert_scivocab_uncased", "allenai/scibert_scivocab_cased"]),
    "biobert": ("BioBERT", ["dmis-lab/biobert-base-cased-v1.2", "dmis-lab/biobert-v1.1",
                             "dmis-lab/biobert-base-cased-v1.1"]),
    "pubmedbert": ("PubMedBERT", ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                                   "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                   "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                                   "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"]),
}
