# AI, NLP & IR Powered Hybrid Product Matching

<img width="1252" height="710" alt="matchingflow" src="https://github.com/user-attachments/assets/8fe19515-eb55-4562-bc66-7b0ee48f6caa" />


This repository contains a robust Python script designed for high-accuracy product matching and deduplication between two separate product catalogs from two different e-commerce platforms. Utilizing techniques from Artificial Intelligence (AI), Natural Language Processing (NLP), and Information Retrieval (IR), Machine Learning the script employs a hybrid approach, combining semantic similarity (Sentence Transformers - S-BERT) for conceptual understanding with lexical matching (BM25) for keyword relevance, and further enhances accuracy using volume and unit validation. The use of FAISS ensures fast and scalable retrieval of candidates.

## Key Features

### Hybrid Scoring Model
Combines the strengths of:

- **Semantic Similarity (S-BERT):** Uses the powerful `BAAI/bge-m3` model to understand the context and meaning of product names.
- **Lexical Matching (BM25):** Utilizes `rank_bm25` to ensure strong matches based on keyword overlap.
- **Volume & Unit Validation:** A sophisticated function `extract_volume` parses complex volume notations (e.g., `4x200ml`, `1.5 Lt`, `500 G`) and applies a bonus or penalty to the final score based on unit volume and pack count consistency.

### FAISS Optimization
Leverages the FAISS library for ultra-fast nearest neighbor search on the S-BERT embeddings, significantly accelerating the matching process over large datasets.

### Dynamic Thresholding
Matches are only finalized if the calculated final hybrid score meets or exceeds a defined `SIMILARITY_THRESHOLD` (set to `0.60` by default).

## Prerequisites

To run this script, you will need the following libraries. It is highly recommended to use a virtual environment.

**Core data processing and utility**
```bash
pip install pandas openpyxl numpy tqdm
```

**NLP and ML components**
```bash
pip install sentence-transformers rank-bm25 scikit-learn
```

**Fast search index**
*Note: Use `faiss-cpu` unless you have specific GPU requirements*
```bash
pip install faiss-cpu
```

## How It Works

The script executes the matching process in the following stages:

1.  **Data Loading and Preprocessing**
    - Loads data from the specified Excel file (`migrosgetir.xlsx`).
    - Cleans product names by converting to lowercase and removing punctuation.
    - The `extract_volume` function is applied to both source and target product names to standardize volumes to a base unit (e.g., all volumes converted to grams or milliliters, depending on the base unit detected).

2.  **Embedding and Indexing**
    - **S-BERT Embedding:** The `BAAI/bge-m3` model encodes all source and target product names into high-dimensional vectors (embeddings).
    - **FAISS Indexing:** The target product embeddings are loaded into a `faiss.IndexFlatL2` structure for fast vector similarity search.
    - **BM25 Setup:** The tokenized target product names are used to build the `BM25Okapi` model.

3.  **Hybrid Matching Loop**
    For every source product:
    - **Candidate Retrieval (FAISS L2):** FAISS is queried to retrieve the Top K (`K=3`) closest semantic candidates from the target catalog. This selection is based on the Euclidean Distance (L2) between the S-BERT embeddings.
    - **Hybrid Score Calculation:** For each candidate, a final score is determined by combining the semantic similarity score (S-BERT) and the lexical matching score (BM25) using a weighted average (60% S-BERT, 40% BM25).
        - **Score SBERT (Semantic Similarity):** Calculated using Cosine Similarity between the source vector and target vector.
        - **Score BM25_Raw (Lexical Matching):** Calculated using the BM25 algorithm (Okapi BM25), which estimates the relevance of documents to a given search query.
        - **Score BM25_Normalized (Normalized BM25):** The raw BM25 score is normalized into the `[0,1]` range using a sigmoid-like function before being used in the hybrid calculation.
    - **Volume Adjustment (Bonus/Penalty):**
        - The unit volume and package counts are compared.
        - A `+0.15` bonus is applied if package counts and unit volumes are highly similar.
        - A penalty (e.g., `-0.2` to `-0.3`) is applied if package counts differ or unit volumes are vastly different, acting as a critical sanity check.
    - **Final Score & Selection:** The score is capped at `1.0`. The candidate with the highest final adjusted score is selected.
    - **Threshold Check:** If the `best_final_score` is `≥0.60`, the match is accepted; otherwise, it is labeled as "Eşleşme Bulunamadı" (No Match Found).
