import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_excel('3d.xlsx')


def preprocess(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


df['clean_productmain'] = df['productmain'].apply(preprocess)
df['clean_productmatch'] = df['productmatch'].apply(preprocess)  # Typo düzeltildi


def extract_volume(text):
    text = text.lower()
    # Geliştirilmiş regex - çoklu paketler, uluslararası birimler ve bitişik yazımlar
    patterns = [
        # Çoklu paketler: 2x100ml, 3 x 250 g
        r'(\d+)\s*x\s*(\d+(?:[\.,]\d+)?)\s*(ml|l|g|kg|oz|lb|fl oz|pcs|pieces|rolls|pack|unit)',
        # Bitişik yazımlar: 1.5lt, 500ml, 2kg
        r'(\d+(?:[\.,]\d+)?)\s*(ml|l|g|kg|oz|lb|fl oz|pcs|pieces|rolls|pack|unit)',
        # Noktalı/virgüllü sayılar: 1,5 L, 2.5 kg
        r'(\d+(?:[\.,]\d+)?)\s*(ml|l|g|kg|oz|lb|fl oz|pcs|pieces|rolls|pack|unit)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if len(match.groups()) == 3:  # Çoklu paket durumu
                quantity = float(match.group(1))
                value = float(match.group(2).replace(',', '.'))
                unit = match.group(3)
                value = quantity * value  # Toplam hacim
            else:  # Normal durum
                value = float(match.group(1).replace(',', '.'))
                unit = match.group(2)

            # Birim dönüşümleri
            if unit in ['l', 'lt']:
                return value * 1000  # Litre -> ml
            elif unit == 'kg':
                return value * 1000  # kg -> g
            elif unit == 'oz':
                return value * 28.35  # oz -> g (yaklaşık)
            elif unit == 'fl oz':
                return value * 29.57  # fl oz -> ml (yaklaşık)
            elif unit == 'lb':
                return value * 453.59  # lb -> g (yaklaşık)
            else:
                return value

    return None


df['volume_productmain'] = df['clean_productmain'].apply(extract_volume)
df['volume_productmatch'] = df['clean_productmatch'].apply(extract_volume)

model = SentenceTransformer('intfloat/multilingual-e5-large')
#model = SentenceTransformer('all-mpnet-base-v2') # english
#model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
#model = SentenceTransformer('sentence-transformers/LaBSE')
#model = SentenceTransformer('BAAI/bge-large-en-v1.5') #english
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')



# Batch encoding - performans iyileştirmesi
print("Source ürünler encode ediliyor...")
source_sbert_embeddings = model.encode(df['clean_productmain'].tolist(), show_progress_bar=True, batch_size=32)
print("Target ürünler encode ediliyor...")
target_sbert_embeddings = model.encode(df['clean_productmatch'].tolist(), show_progress_bar=True, batch_size=32)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# Source ve target için ayrı TF-IDF
all_texts = df['clean_productmain'].tolist() + df['clean_productmatch'].tolist()
vectorizer.fit(all_texts)
source_tfidf = vectorizer.transform(df['clean_productmain'])
target_tfidf = vectorizer.transform(df['clean_productmatch'])

# Performans iyileştirmesi: Tüm benzerlikler bir kerede hesaplanıyor
print("SBERT benzerlikleri hesaplanıyor...")
sbert_similarities = cosine_similarity(source_sbert_embeddings, target_sbert_embeddings)
print("TF-IDF benzerlikleri hesaplanıyor...")
tfidf_similarities = cosine_similarity(source_tfidf, target_tfidf)

results = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Eşleştirme yapılıyor"):
    source_volume = row['volume_productmain']
    sbert_sims = sbert_similarities[i]
    tfidf_sims = tfidf_similarities[i]
    hybrid_scores = 0.6 * sbert_sims + 0.4 * tfidf_sims

    top3_idx = np.argsort(hybrid_scores)[-3:][::-1]  # En yüksekten 3 skor

    top_matches = []
    for idx in top3_idx:
        base_score = hybrid_scores[idx]
        target_volume = df.iloc[idx]['volume_productmatch']

        # Volume bonus hesaplama
        if source_volume and target_volume:
            source_text = str(row['productmain']) if pd.notna(row['productmain']) else ""
            target_text = str(df.iloc[idx]['productmatch']) if pd.notna(df.iloc[idx]['productmatch']) else ""

            source_pack_match = re.search(r'(\d+)\s*x', source_text.lower())
            target_pack_match = re.search(r'(\d+)\s*x', target_text.lower())

            source_pack_count = int(source_pack_match.group(1)) if source_pack_match else 1
            target_pack_count = int(target_pack_match.group(1)) if target_pack_match else 1

            source_unit_volume = source_volume / source_pack_count
            target_unit_volume = target_volume / target_pack_count
            unit_volume_diff = abs(source_unit_volume - target_unit_volume) / max(source_unit_volume, target_unit_volume)

            if source_pack_count == target_pack_count and unit_volume_diff < 0.2:
                volume_bonus = 0.15
            elif source_pack_count != target_pack_count:
                volume_bonus = -0.2
            elif unit_volume_diff > 0.5:
                volume_bonus = -0.3
            else:
                volume_bonus = 0
        else:
            volume_bonus = 0

        final_score = min(base_score + volume_bonus, 1.0)
        top_matches.append((
            df.iloc[idx]['productmatch'],
            df.iloc[idx]['productcode'],
            f"{round(final_score * 100, 2)}%"
        ))

    results.append({
        'productmain': row['productmain'],
        'match_1': top_matches[0][0],
        'code_1': top_matches[0][1],
        'score_1': top_matches[0][2],
        'match_2': top_matches[1][0],
        'code_2': top_matches[1][1],
        'score_2': top_matches[1][2],
        'match_3': top_matches[2][0],
        'code_3': top_matches[2][1],
        'score_3': top_matches[2][2],
    })

match_df = pd.DataFrame(results)
match_df.to_excel('e5sim*100%.xlsx', index=False)
print("done")

print("Source embeddings boyutu:", source_sbert_embeddings.shape)
print("Target embeddings boyutu:", target_sbert_embeddings.shape)

##Performans sorunu - O(n²) karmaşıklık Batch encoding eksikliği Hacim regex sınırlılıkları Typo hatası