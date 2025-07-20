# NLP Product Matching Algorithm

This project provides a hybrid NLP model to match product listings from different data sources. It combines semantic similarity (SBERT) and keyword-based matching (TF-IDF) with rule-based logic for features like volume and weight to achieve high accuracy.

## Features

- **Hybrid Similarity:** Merges SBERT and TF-IDF scores for more robust text matching.
- **Volume & Weight Parsing:** Uses regex to extract and compare product sizes (e.g., "2x500ml", "1.5 kg"), adjusting the final score based on this data.
- **Optimized for Performance:** Employs batch processing for embeddings and similarity calculations to handle large datasets efficiently.
- **Multiple Modes:** Includes scripts to find either the single best match or the top 3 potential matches for each product.

## Technologies

- Python
- Pandas
- Scikit-learn
- Sentence-Transformers
- Numpy

## How It Works

The matching process follows four main steps:

1.  **Preprocessing:** Product names are cleaned by converting them to lowercase and removing punctuation.
2.  **Feature Extraction:** Volume, weight, and pack size are parsed from the text using regular expressions. Units are standardized (e.g., L to mL) for accurate comparison.
3.  **Similarity Calculation:** A weighted hybrid score is calculated from the cosine similarities of SBERT (semantic) and TF-IDF (keyword) vectors.
4.  **Score Adjustment:** The initial score is fine-tuned based on the volume/weight comparison. A bonus is applied for closely matching volumes, and a penalty is applied for significant differences, leading to a more precise final score.
