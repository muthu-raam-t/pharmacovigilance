# Pharmacovigilance — Joint BioBERT Framework for Drug Safety

## Overview

This project builds a multi-task deep learning system for automated
pharmacovigilance — detecting drug-disease and drug-side-effect relationships
directly from biomedical text. A shared transformer backbone is jointly
fine-tuned with a named entity recognition (NER) head and a relation
extraction (RE) head, so entity detection and relationship classification
learn from shared context instead of running as disconnected stages.

The extracted relationships will be fused with curated structured safety
databases (SIDER, OnSIDES, DrugBank) into a knowledge graph, ranked by an
evidence-based scoring engine, and explained through an explainable AI layer —
ultimately surfaced through an advisory interface where a user can look up
either a disease (to get ranked drug recommendations) or a drug (to get its
side-effect profile and safer alternatives).

A core part of this project is a fair, controlled comparison across 5
transformer backbones — BERT-base, RoBERTa, SciBERT, PubMedBERT, and BioBERT —
all sharing the identical NER+RE architecture and training procedure, isolating
pretraining domain as the one variable that changes between runs.

## Completed so far

- Unified preprocessing pipeline merging three biomedical text corpora
  (BC5CDR, BioRED, ADE Corpus) into one consistent schema with normalized
  entity and relation labels.
- Structured knowledge base (SIDER, OnSIDES) loaded into PostgreSQL.
- Classical TF-IDF + Linear SVM baseline, used to validate dataset quality
  before transformer training.
- Shared joint NER+RE model architecture, reusable across any transformer
  backbone.
- BERT-base and BioBERT trained and evaluated on the held-out test set.
