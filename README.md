# security-log-augmentation-framework

## Security Log Augmentation

This repository contains the core implementation accompanying the paper:

**Semantics-Preserving Security Log Augmentation for Robust Machine Learning in SIEM Systems**

The project provides a domain-aware augmentation framework for security logs. Unlike generic text augmentation, the framework treats security logs as semi-structured operational records whose meaning depends on typed entities, field relationships, source-specific syntax, and character-level annotation consistency.

## Overview

The framework is designed to generate diverse training data for AI-supported security-log processing while preserving semantic and structural validity. It operates on span-aware log records and uses a standardized model of security-specific metakeys to control which values may be modified and how.

The main components include:

- a standardized metakey model for security-relevant log attributes,
- metakey-guided augmentation policies,
- span-aware representation of annotated log records,
- value-level and record-level augmentation operators,
- pattern-based, list-based, and neural contextual value generation,
- validation mechanisms for preserving annotation alignment.

## Research Context

The implementation was developed as part of research on AI-supported security monitoring and SIEM-oriented log processing. The framework was evaluated on a Named Entity Recognition (NER) task with exact span matching, where NER is treated as a prerequisite for downstream SIEM-related processing such as parsing, normalization, enrichment, and correlation.

---

## NCG Component

This module implements the **neural contextual generation (NCG)** component of the framework. It fine-tunes transformer-based language models on security log datasets and uses them to fill masked entity spans in log records.

## Structure

```
security-log-augmentation-framework/
├── pipeline/
│   ├── pipeline.py               # Main DeepAugmentator class — loads model from MLflow, runs augmentation
│   ├── AI_augmentator.py         # Fill-mask augmentation (MLM models: RoBERTa, ALBERT, MobileBERT, ELECTRA)
│   └── AI_Ollama_augmentator.py  # Next-word prediction via Ollama API (LLaMA, DeepSeek)
├── train_scripts/
│   ├── MLM_training.py           # Fine-tuning for Masked Language Models
│   ├── gpt_training.py           # Fine-tuning for GPT-2
│   ├── llama_training.py         # Fine-tuning for LLaMA 3.2
│   ├── smollm2_instruct_training.py  # Fine-tuning for SmolLM2
│   └── run.sh                    # Training launcher — select model and dataset path here
└── MLflow_model_load/
    └── MLM_models.ipynb          # Notebook for loading and testing models from MLflow registry
```

## How it works

Log records with `<mask>` tokens in entity spans are passed to the model. The model predicts a replacement value from the surrounding log context. Entity offsets are recalculated after substitution.

Two generation modes are supported:
- **MLM (fill-mask)** — bidirectional context via RoBERTa, ALBERT, MobileBERT, ELECTRA
- **NWP (next-word prediction)** — autoregressive generation via Ollama API (LLaMA 3.2, DeepSeek)

## Training

Edit `run.sh` to select a model and dataset path, then run:

```bash
cd train_scripts
bash run.sh
```

Trained models are saved to `MLM_trained_models/<model>/<version>/` and tracked in MLflow.

## Inference

```python
from pipeline.pipeline import DeepAugmentator

augmentator = DeepAugmentator(
    model_path="path/to/model",
    tokenizer_path="path/to/tokenizer",
    keep_mask=True
)
```

Or load directly from MLflow model registry — see `MLflow_model_load/MLM_models.ipynb`.

## Dependencies

- `transformers`, `torch`, `datasets`, `mlflow`, `ollama`
