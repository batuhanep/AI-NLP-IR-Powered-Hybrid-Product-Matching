import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss


# --- hacim
def extract_volume(text):
    text = str(text).lower()
    patterns = [
        r'(\d+)\s*(adet|pcs|pieces|tane|units|unit|kutu|paket|şişe|poşet|kavanoz|takım|çift|rulo|koli|pack|box|bottle|can|jar|roll|dozen|case|package|count|set|pair)', r'(\d+)\s*\'\s*(li|lı|lü|lu)\s*(paket|kutu|koli|takım|set|şişe|poşet)?',
        r'(\d+)\s*[x*]\s*(\d+(?:[\.,]\d+)?)\s*(/|in|inç|inch|ekran|btu|hz|ms|devir|rpm|tb|gb|mb|cl|g|gr|gm|mg|ml|l|lt|cc|kg|oz|lb|fl oz|gram|kilogram|litre|liter|mililitre|milliliter|m|cm|mm|metre|meter|santimetre|centimeter|w|v|mah|kw|kcal)',
        r'(\d+(?:[\.,]\d+)?)\s*-\s*\d+(?:[\.,]\d+)?\s*(/|in|inç|inch|ekran|btu|hz|ms|devir|rpm|tb|gb|mb|cl|g|gr|gm|mg|ml|l|lt|cc|kg|oz|lb|fl oz|gram|kilogram|litre|liter|mililitre|milliliter|m|cm|mm|metre|meter|santimetre|centimeter|w|v|mah|kw|kcal)',
        r'(\d+(?:[\.,]\d+)?)\s*(/|in|inç|inch|ekran|btu|hz|ms|devir|rpm|tb|gb|mb|cl|g|gr|gm|mg|ml|l|lt|cc|kg|oz|lb|fl oz|gram|kilogram|litre|liter|mililitre|milliliter|m|cm|mm|metre|meter|santimetre|centimeter|w|v|mah|kw|kcal)'
    ]
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if match:
            if i < 2: continue
            if len(match.groups()) == 3 and match.group(3) is not None:
                quantity = float(match.group(1));
                value = float(match.group(2).replace(',', '.'));
                unit = match.group(3)
                value = quantity * value
            else:
                value = float(match.group(1).replace(',', '.'));
                unit = match.group(2)
            if unit in ['l', 'lt', 'litre', 'liter']:
                return value * 1000
            elif unit in ['kg', 'kilogram']:
                return value * 1000
            elif unit == 'oz':
                return value * 28.35
            elif unit == 'fl oz':
                return value * 29.57
            elif unit == 'lb':
                return value * 453.59
            elif unit in ['m', 'metre', 'meter']:
                return value * 100
            elif unit == 'in':
                return value * 2.54
            else:
                return value
    return None


# --- ön işleme
df = pd.read_excel('/Users/batuhanozdemir/Desktop/datasicence-ops-software/Datascience/getir.xlsx')


def preprocess(text):
    if pd.isnull(text): return ""
    text = str(text).lower();
    text = re.sub(r'[^\w\s.,]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


df['volume_productmain'] = df['productmain'].apply(extract_volume)
df['volume_productmatch'] = df['productmatch'].apply(extract_volume)

df['clean_productmain'] = df['productmain'].apply(preprocess)
df['clean_productmatch'] = df['productmatch'].apply(preprocess)

# --- model
model = SentenceTransformer('BAAI/bge-m3')

print("Source ürünler encode ediliyor...")
source_sbert_embeddings = model.encode(df['clean_productmain'].tolist(), show_progress_bar=True, batch_size=32).astype(
    'float32')
print("Target ürünler encode ediliyor...")
target_sbert_embeddings = model.encode(df['clean_productmatch'].tolist(), show_progress_bar=True, batch_size=32).astype(
    'float32')

# bm25
print("BM25 için metinler tokenize ediliyor...")
tokenized_source = [doc.split(" ") for doc in df['clean_productmain'].tolist()]
tokenized_target = [doc.split(" ") for doc in df['clean_productmatch'].tolist()]

# Aranacak hedef metinler ile BM25 modelini oluşturuyoruz.
print("BM25 modeli oluşturuluyor...")
bm25 = BM25Okapi(tokenized_target)
# --------------------------

# --- faiss hızlı arama endeksi
print("FAISS indeksi oluşturuluyor...")
embedding_size = target_sbert_embeddings.shape[1]
# L2, öklid mesafesi hesaplama
index_sbert = faiss.IndexFlatL2(embedding_size)
# aranacak ürünlerin sbert embedding'lerini indekse ekliyoruz
index_sbert.add(target_sbert_embeddings)
print(f"{index_sbert.ntotal} adet hedef ürün indekse eklendi.")

# eşik değeri belirleme
results = []

SIMILARITY_THRESHOLD = 0.60

for i, row in tqdm(df.iterrows(), total=len(df), desc="Eşleştirme yapılıyor"):
    # embeddings
    source_sbert_vector = source_sbert_embeddings[i:i + 1]  # 2d dizin
    # BM25 için tokenized sorgusu
    source_bm25_query = tokenized_source[i]
    source_volume = row['volume_productmain']

    # ----------------------------------------------------
    # BM25 skorlarını hesapla
    bm25_scores_all = bm25.get_scores(source_bm25_query)

    # Retrieval(aday belirleme)

    # A) FAISS ile Semantik Adaylar Anlam benzerliği
    K_SEMANTIC = 10   
    distances, indices = index_sbert.search(source_sbert_vector, K_SEMANTIC) 
    faiss_candidates = indices[0].tolist()

    # B) BM25 ile Kelime/Kod Bazlı Adaylar 
    K_KEYWORD = 10
    # en yüksek 10 indeksin listesi
    bm25_candidates = np.argsort(bm25_scores_all)[-K_KEYWORD:][::-1].tolist()

    # tüm adayları birleştir
    candidate_indices = list(set(faiss_candidates + bm25_candidates))
    
    candidates_with_scores = [] 

    #  Skorlama
    for candidate_idx in candidate_indices:
        # geçersiz indeksleri yoksay
        if candidate_idx == -1:
            continue

        # SBERT Skoru 
        sbert_score = cosine_similarity(source_sbert_vector, target_sbert_embeddings[candidate_idx:candidate_idx + 1])[0][0]

        # BM25 Skoru 
        raw_bm25_score = bm25_scores_all[candidate_idx]

        # bm25 skorlarını normalleştirme adımı.
        normalized_bm25_score = 1 - (1 / (1 + raw_bm25_score))  # sigmoid function
        
        hybrid_score = 0.6 * sbert_score + 0.4 * normalized_bm25_score

        #  hacim bonus/ceza mantığı
        target_volume = df.iloc[candidate_idx]['volume_productmatch']
        volume_bonus = 0
        if source_volume and target_volume:
            # hacim kontrolü
            source_text = str(row['productmain']) if pd.notna(row['productmain']) else ""
            target_text = str(df.iloc[candidate_idx]['productmatch']) if pd.notna(
                df.iloc[candidate_idx]['productmatch']) else ""

            source_pack_match = re.search(r'(\d+)\s*x', source_text.lower())
            target_pack_match = re.search(r'(\d+)\s*x', target_text.lower())

            source_pack_count = int(source_pack_match.group(1)) if source_pack_match else 1
            target_pack_count = int(target_pack_match.group(1)) if target_pack_match else 1

            # birim hacim
            source_unit_volume = source_volume / source_pack_count
            target_unit_volume = target_volume / target_pack_count

            # hacim farkı?
            unit_volume_diff = abs(source_unit_volume - target_unit_volume) / max(source_unit_volume,
                                                                                  target_unit_volume)

            # final skor
            if source_pack_count == target_pack_count and unit_volume_diff < 0.2:
                volume_bonus = 0.15
            elif source_pack_count != target_pack_count:
                volume_bonus = -0.2
            elif unit_volume_diff > 0.5:
                volume_bonus = -0.3
            else:
                volume_bonus = 0

        final_score = min(hybrid_score + volume_bonus, 1.0)

        # adayı listeye ekle
        candidates_with_scores.append({
            'product': df.iloc[candidate_idx]['productmatch'],
            'code': df.iloc[candidate_idx]['productcode'],
            'score': final_score
        })

    # -------

    # puanı en yüksekten düşüğe sırala
    candidates_with_scores.sort(key=lambda x: x['score'], reverse=True)

    # sonuç 
    row_result = {'productmain': row['productmain']}
    
    # ilk 3 adayı yan yana yazar
    for rank in range(3):
        suffix = rank + 1
        if rank < len(candidates_with_scores) and candidates_with_scores[rank]['score'] > SIMILARITY_THRESHOLD:
            c = candidates_with_scores[rank]
            # Burada threshold kontrolü istersen if c['score'] > 0.60 ekleyebilirsin
            row_result[f'match_{suffix}_product'] = c['product']
            row_result[f'match_{suffix}_code'] = c['code']
            row_result[f'match_{suffix}_score'] = f"%{c['score']*100:.2f}"
        else:
            row_result[f'match_{suffix}_product'] = "N/A"
            row_result[f'match_{suffix}_code'] = "N/A"
            row_result[f'match_{suffix}_score'] = "N/A"

    results.append(row_result)

# output
match_df = pd.DataFrame(results)
match_df.to_excel('getirfinal.xlsx', index=False)
print("done")

