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
surfaced through an advisory interface where a user looks up either a disease
(to get ranked drug recommendations) or a drug (to get its side-effect profile
and safer alternatives).

## Model comparison

A core contribution of this project is a fair, controlled comparison across
5 transformer backbones, all sharing the identical NER+RE architecture and
training procedure — isolating pretraining domain as the one variable that
changes between runs.

- **BERT-base** — general-purpose baseline, pretrained on Wikipedia and
  BookCorpus, no biomedical exposure. Establishes the lower bound for
  transformer performance on this task.
- **RoBERTa** — general-purpose, trained with a larger corpus and refined
  pretraining recipe than BERT. Tests whether a stronger general-domain
  model can close the gap without biomedical data.
- **SciBERT** — pretrained on scientific literature broadly, not biomedical
  specifically. Tests a partial domain shift.
- **PubMedBERT** — pretrained from scratch on biomedical text (PubMed
  abstracts and full text), no general-domain pretraining at all.
- **BioBERT** — the proposed model, continued-pretrained from general BERT
  on PubMed and PMC full-text articles, combining general language
  understanding with biomedical specialization.

## Results

Evaluated on an identical held-out test set, same architecture, same training
configuration (3 epochs, batch size 8) across all 5 models.

![Model comparison table](data/processed/metrics_images/all_models_comparison_table.png)

## Completed so far

- Unified preprocessing pipeline merging three biomedical text corpora
  (BC5CDR, BioRED, ADE Corpus) into one consistent schema with normalized
  entity and relation labels.
- Structured knowledge base (SIDER, OnSIDES) loaded into PostgreSQL.
- Classical TF-IDF + Linear SVM baseline, used to validate dataset quality
  before transformer training.
- Shared joint NER+RE model architecture, reusable across any transformer
  backbone.
- All 5 comparison models (BERT-base, RoBERTa, SciBERT, PubMedBERT, BioBERT)
  trained and evaluated on the held-out test set, with results visualized.
