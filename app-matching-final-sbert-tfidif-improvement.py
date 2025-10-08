import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss

# --- hacim
def extract_volume(text):
    text = str(text).lower()
    patterns = [
        r'(\d+)\s*(adet|pcs|pieces|tane|units|unit)', r'(\d+)\s*\'\s*(li|lı)\s*(paket)?',
        r'(\d+)\s*x\s*(\d+(?:[\.,]\d+)?)\s*(g|gm|mg|ml|l|kg|oz|lb|fl oz)',
        r'(\d+(?:[\.,]\d+)?)\s*-\s*\d+(?:[\.,]\d+)?\s*(g|gm|mg|ml|l|kg|oz|lb|fl oz)',
        r'(\d+(?:[\.,]\d+)?)\s*(g|gm|mg|ml|l|kg|oz|lb|fl oz)'
    ]
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if match:
            if i < 2: continue
            if len(match.groups()) == 3 and match.group(3) is not None:
                quantity = float(match.group(1)); value = float(match.group(2).replace(',', '.')); unit = match.group(3)
                value = quantity * value
            else:
                value = float(match.group(1).replace(',', '.')); unit = match.group(2)
            if unit in ['l', 'lt']: return value * 1000
            elif unit == 'kg': return value * 1000
            elif unit == 'oz': return value * 28.35
            elif unit == 'fl oz': return value * 29.57
            elif unit == 'lb': return value * 453.59
            else: return value
    return None

# --- ön işleme
df = pd.read_excel('ip16.xlsx')
def preprocess(text):
    if pd.isnull(text): return ""
    text = str(text).lower(); text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

df['clean_productmain'] = df['productmain'].apply(preprocess)
df['clean_productmatch'] = df['productmatch'].apply(preprocess)
df['volume_productmain'] = df['clean_productmain'].apply(extract_volume)
df['volume_productmatch'] = df['clean_productmatch'].apply(extract_volume)

# --- model
model = SentenceTransformer('BAAI/bge-m3')
print("Source ürünler encode ediliyor...")
source_sbert_embeddings = model.encode(df['clean_productmain'].tolist(), show_progress_bar=True, batch_size=32).astype('float32')
print("Target ürünler encode ediliyor...")
target_sbert_embeddings = model.encode(df['clean_productmatch'].tolist(), show_progress_bar=True, batch_size=32).astype('float32')

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
all_texts = df['clean_productmain'].tolist() + df['clean_productmatch'].tolist()
vectorizer.fit(all_texts)
source_tfidf = vectorizer.transform(df['clean_productmain'])
target_tfidf = vectorizer.transform(df['clean_productmatch'])


# --- faiss hızlı arama endeksi
print("FAISS indeksi oluşturuluyor...")
embedding_size = target_sbert_embeddings.shape[1]
# IndexFlatL2, standart ve basit bir indekstir. Vektörler arası mesafeyi (L2) hesaplar.
index_sbert = faiss.IndexFlatL2(embedding_size)
# Sadece hedef (aranacak) ürünlerin SBERT embedding'lerini indekse ekliyoruz.
index_sbert.add(target_sbert_embeddings)
print(f"{index_sbert.ntotal} adet hedef ürün indekse eklendi.")


# eşik değeri belirleme
results = []

SIMILARITY_THRESHOLD = 0.60

for i, row in tqdm(df.iterrows(), total=len(df), desc="Eşleştirme yapılıyor"):
    # embeddings
    source_sbert_vector = source_sbert_embeddings[i:i+1] # FAISS 2D dizi bekler
    source_tfidf_vector = source_tfidf[i]
    source_volume = row['volume_productmain']

    # FAISS ile en yakın 1 adayı bul
    # D: Mesafe (düşük olan daha iyi), I: İndeks (en iyi adayın satır numarası)
    distances, indices = index_sbert.search(source_sbert_vector, 1)
    best_idx = indices[0][0] # En iyi adayın indeksi

    # TF-IDF ve SBERT skorlarını hesapla
    # Not: FAISS L2 mesafesi döndürür, cosine similarity değil. Skoru çevirebiliriz ama
    # hibrit skor için direkt cosine similarity daha sezgiseldir.
    # Bu yüzden sadece en iyi adayla olan benzerliği tekrar hesaplıyoruz. Bu çok hızlıdır.
    sbert_score = cosine_similarity(source_sbert_vector, target_sbert_embeddings[best_idx:best_idx+1])[0][0]
    tfidf_score = cosine_similarity(source_tfidf_vector, target_tfidf[best_idx])[0][0]

    # Hibrit skor
    hybrid_score = 0.6 * sbert_score + 0.4 * tfidf_score

    # Hacim bonus/ceza mantığı
    target_volume = df.iloc[best_idx]['volume_productmatch']
    volume_bonus = 0.1 if source_volume and target_volume and abs(source_volume - target_volume) <= 10 else 0
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
    final_score = min(hybrid_score + volume_bonus, 1.0)

    # eşik değeri
    if final_score >= SIMILARITY_THRESHOLD:
        # Eşiği geçerse, eşleşmeyi kaydet
        results.append({
            'productmain': row['productmain'],
            'matched_product': df.iloc[best_idx]['productmatch'],
            'productcode': df.iloc[best_idx]['productcode'],
            'similarity_score': f"{round(final_score * 100, 2)}%"
        })
    else:
        # eşiği geçmezse eşleşme yok
        results.append({
            'productmain': row['productmain'],
            'matched_product': 'Eşleşme Bulunamadı',
            'productcode': 'N/A',
            'similarity_score': f"{round(final_score * 100, 2)}%"
        })

# output
match_df = pd.DataFrame(results)
match_df.to_excel('ip16f.xlsx', index=False)
print("done")