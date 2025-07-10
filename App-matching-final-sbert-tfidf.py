import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_excel('yoyo.xlsx')


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

#model = SentenceTransformer('intfloat/multilingual-e5-large')
#model = SentenceTransformer('all-mpnet-base-v2') # english
#model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
model = SentenceTransformer('sentence-transformers/LaBSE')
#model = SentenceTransformer('BAAI/bge-large-en-v1.5') #english


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

    # Önceden hesaplanmış benzerliklerden al
    sbert_sims = sbert_similarities[i]
    tfidf_sims = tfidf_similarities[i]

    hybrid_scores = 0.6 * sbert_sims + 0.4 * tfidf_sims
    best_idx = hybrid_scores.argmax()
    best_score = hybrid_scores[best_idx]

    target_volume = df.iloc[best_idx]['volume_productmatch']
#    volume_bonus = 0.1 if source_volume and target_volume and abs(source_volume - target_volume) <= 10 else 0
    # Paket kontrolü eklenmiş volume_bonus hesaplaması
    if source_volume and target_volume:
        # Paket sayılarını kontrol et - STR DÖNÜŞÜMÜ EKLENDİ
        source_text = str(row['productmain']) if pd.notna(row['productmain']) else ""
        target_text = str(df.iloc[best_idx]['productmatch']) if pd.notna(df.iloc[best_idx]['productmatch']) else ""

        source_pack_match = re.search(r'(\d+)\s*x', source_text.lower())
        target_pack_match = re.search(r'(\d+)\s*x', target_text.lower())

        source_pack_count = int(source_pack_match.group(1)) if source_pack_match else 1
        target_pack_count = int(target_pack_match.group(1)) if target_pack_match else 1

        # Birim hacim hesapla
        source_unit_volume = source_volume / source_pack_count
        target_unit_volume = target_volume / target_pack_count

        # Hacim farkı hesapla
        unit_volume_diff = abs(source_unit_volume - target_unit_volume) / max(source_unit_volume, target_unit_volume)

        # Skorlama
        if source_pack_count == target_pack_count and unit_volume_diff < 0.2:
            volume_bonus = 0.15  # Aynı paket sayısı + yakın hacim
        elif source_pack_count != target_pack_count:
            volume_bonus = -0.2  # Farklı paket sayısı cezası
        elif unit_volume_diff > 0.5:
            volume_bonus = -0.3  # Çok farklı hacim
        else:
            volume_bonus = 0
    else:
        volume_bonus = 0
    final_score = min(best_score + volume_bonus, 1.0)

    results.append({
        'productmain': row['productmain'],
        'matched_product': df.iloc[best_idx]['productmatch'],  # Typo düzeltildi
        'productcode': df.iloc[best_idx]['productcode'],
        'similarity_score': final_score
    })

match_df = pd.DataFrame(results)
match_df.to_excel('yoyofffdde5.xlsx', index=False)
print("done")


##Performans sorunu - O(n²) karmaşıklık Batch encoding eksikliği Hacim regex sınırlılıkları Typo hatası