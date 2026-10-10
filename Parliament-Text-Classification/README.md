# Parliament Petitions Text Classification (GovTriage-NLP)

## Project Overview
This project tackles an industrial Machine Learning Engineering (MLE) problem: **Automated Document Routing**. It classifies public petitions submitted to the UK Parliament and predicts which government department (e.g., Department of Health, Home Office) should handle them. This represents a highly novel domain in Natural Language Processing (Legal/Political NLP).

## Evaluation Metric
We selected **Macro F1-Score** as our primary evaluation metric alongside Accuracy. 
**Why Macro F1?** Real-world petition data suffers from extreme class imbalance (e.g., the Department of Health receives significantly more petitions than the Department for Transport). Accuracy can be misleading by favoring the majority class. Macro F1 treats all classes equally, ensuring our model performs robustly across both popular and rare departments.

## Repository Structure
- `data/raw/`: Contains the original JSON-derived data fetched from the UK Parliament API (`parliament_petitions_raw.csv`).
- `data/cleaned/`: Contains the noise-free dataset after preprocessing (`parliament_petitions_clean.csv`).
- `src/1_data_scraper.py`: Fetches real-time public petition data via the official UK Parliament API.
- `src/2_preprocess.py`: Our data preprocessing pipeline. It handles missing values, removes URLs, punctuation, and non-alphanumeric characters using Regex, and filters out extreme minority classes to combat class imbalance.
- `src/3_train.py`: The traditional training script that vectorizes text using TF-IDF and trains our Logistic Regression and Linear SVC models.
- `src/3_advanced_train.py`: **[V2 Feature]** The advanced Deep Learning pipeline. It trains the Vanilla Baseline (BERT) and runs the JEV (LLM-as-a-judge) simulation.
- `contributions/`: Contains individual markdown files detailing team member contributions.

## Preprocessing Pipeline
1. **Lowercasing & Noise Removal:** Converted all text to lowercase. Used regular expressions (`re`) to strip emojis, punctuation, and special characters.
2. **Whitespace Normalization:** Reduced multiple spaces and line breaks into single spaces.
3. **Class Filtering:** Filtered the dataset to retain only the top 5 most frequently assigned departments to allow the model to learn effectively without extreme rare-class interference.

## Model Benchmarks & Advanced JEV Evaluation

Per the course guidelines, we separated our architecture and model versions. We trained our Deep Learning models using **Apple Silicon (M2 MPS - Metal Performance Shaders)** for hardware acceleration. We also implemented a **JEV (LLM-as-a-judge) Alternative** using an open-source Zero-Shot classifier to compare traditional fine-tuning vs. modern prompt-based routing.

| Architecture | Model Variant | Approach | Macro F1-Score |
| :--- | :--- | :--- | :--- |
| **Traditional ML** | Linear SVC | TF-IDF Vectorization | 0.8768 |
| **Transformer (Vanilla Baseline)** | BERT-base-uncased | Fine-Tuned on Dataset | **0.9346 (Winner)** |
| **LLM-as-a-judge (JEV Alt.)** | BART-large-mnli | Zero-Shot Classification | 0.3412 |

*Conclusion: The fine-tuned Vanilla Baseline (BERT-base-uncased) significantly outperformed both the traditional TF-IDF approaches and the Zero-Shot LLM (JEV) alternative, proving the necessity of task-specific training for domain-heavy political texts.*

## How to Run
1. Install requirements: `pip install -r requirements.txt` (Ensure `accelerate>=1.1.0` is installed for Mac MPS support).
2. Fetch data: `python src/1_data_scraper.py`
3. Clean data: `python src/2_preprocess.py`
4. Train Traditional Models: `python src/3_train.py`
5. Train BERT & Run JEV Simulation: `python src/3_advanced_train.py`