import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel('dosya_adı.xlsx')

def preprocess(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

df['clean_productmain'] = df['productmain'].apply(preprocess)
df['clean_productmatch'] = df['productmacth'].apply(preprocess)

def extract_volume(text):
    text = text.lower()
    match = re.search(r'(\d+(?:[\.,]\d+)?)\s*(ml|l|g|kg)', text)
    if not match:
        return None
    value = float(match.group(1).replace(',', '.'))
    unit = match.group(2)
    return value * 1000 if unit in ['l', 'kg'] else value

df['volume_productmain'] = df['clean_productmain'].apply(extract_volume)
df['volume_productmatch'] = df['clean_productmatch'].apply(extract_volume)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
target_sbert_embeddings = model.encode(df['clean_productmatch'].tolist(), show_progress_bar=True)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
target_tfidf = vectorizer.fit_transform(df['clean_productmatch'])

results = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Multilingual eşleştirme"):
    source_text = row['clean_productmain']
    source_volume = row['volume_productmain']

    sbert_vec = model.encode([source_text])
    sbert_sims = cosine_similarity(sbert_vec, target_sbert_embeddings).flatten()

    tfidf_vec = vectorizer.transform([source_text])
    tfidf_sims = cosine_similarity(tfidf_vec, target_tfidf).flatten()

    hybrid_scores = 0.6 * sbert_sims + 0.4 * tfidf_sims
    best_idx = hybrid_scores.argmax()
    best_score = hybrid_scores[best_idx]

    target_volume = df.iloc[best_idx]['volume_productmatch']
    volume_bonus = 0.1 if source_volume and target_volume and abs(source_volume - target_volume) <= 10 else 0

    final_score = min(best_score + volume_bonus, 1.0)

    results.append({
        'productmain': row['productmain'],
        'matched_product': df.iloc[best_idx]['productmacth'],
        'productcode': df.iloc[best_idx]['productcode'],
        'similarity_score': final_score
    })

match_df = pd.DataFrame(results)
match_df.to_excel('hibrit_multilingual_sonuclar.xlsx', index=False)
print("✅ Multilingual eşleştirme tamamlandı: hibrit_multilingual_sonuclar.xlsx")
