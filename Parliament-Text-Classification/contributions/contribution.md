# Team Contributions

This document outlines the specific contributions of each team member to the **GovTriage-NLP (Automated Document Routing)** project.

## Adnan Burak Özbay (2309111073)
**Role: Project Lead & Advanced Modeling**
* Designed the core code architecture and repository structure.
* Configured Apple Silicon (M2 MPS) hardware acceleration to run deep learning models locally.
* Integrated the `BERT-base-uncased` (Vanilla Baseline) model pipeline, achieving the winning score of 93.46% Macro F1.
* Developed and implemented the Zero-Shot LLM (`bart-large-mnli`) pipeline to serve as the JEV (LLM-as-a-judge) alternative.

## Enes Erman (2309111066)
**Role: Data Engineering & Preprocessing**
* Developed the `1_data_scraper.py` script to fetch 3,000+ raw petitions directly from the official UK Parliament JSON API, securing the Data Novelty requirement.
* Built the text preprocessing pipeline (`2_preprocess.py`) to remove noise, URLs, and punctuation using Regular Expressions (Regex).
* Implemented the Top-5 department filtering logic to handle extreme class imbalance in the dataset.

## İlayda Şenel (2309011159)
**Role: Traditional ML Modeling & Evaluation Strategy**
* Handled the feature extraction process by implementing TF-IDF Vectorization to penalize common words and emphasize domain-specific vocabulary.
* Trained, tuned, and evaluated the traditional baseline models (Logistic Regression and Linear SVC).
* Formulated the evaluation strategy by strictly selecting the **Macro F1-Score** over Accuracy to prevent misleading results caused by imbalanced classes.

## Mohammed Firas (2309015722)
**Role: Presentation Strategy & Technical Documentation**
* Designed the final presentation architecture, ensuring a balance between high technical depth and visual simplicity according to the rubric constraints.
* Authored and formatted the final `README.md` documentation.
* Structured the model evaluation tables, specifically ensuring the strict distinction between Model Architectures (e.g., Transformer vs. Traditional ML) and Model Variants (e.g., BERT-base).