import pandas as pd
import numpy as np
import re
import faiss
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --- 1. Konfigürasyon ve Loglama ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    source_path: str = '/Users/batuhanozdemir/Desktop/datasicence-ops-software/Datascience/gtr11.xlsx'
    output_path: str = 'outputgtr112.xlsx'
    model_name: str = 'BAAI/bge-m3'
    similarity_threshold: float = 0.60
    top_k: int = 10
    batch_size: int = 32

# --- 2. Yardımcı Sınıflar (Preprocessing) ---
# --- BU SINIFI SENİN ORİJİNAL REGEX'İN İLE GÜNCELLİYORUZ ---
class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        # Senin orijinal preprocess fonksiyonun
        if pd.isnull(text): return ""
        text = str(text).lower();
        text = re.sub(r'[^\w\s.,]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def extract_volume(text):
        # --- SENİN ORİJİNAL KODUNU BURAYA YAPIŞTIRDIM ---
        text = str(text).lower()
        patterns = [
            r'(\d+)\s*(adet|pcs|pieces|tane|units|unit|kutu|paket|şişe|poşet|kavanoz|takım|çift|rulo|koli|pack|box|bottle|can|jar|roll|dozen|case|package|count|set|pair)', 
            r'(\d+)\s*\'\s*(li|lı|lü|lu)\s*(paket|kutu|koli|takım|set|şişe|poşet)?',
            r'(\d+)\s*[x*]\s*(\d+(?:[\.,]\d+)?)\s*(/|in|inç|inch|ekran|btu|hz|ms|devir|rpm|tb|gb|mb|cl|g|gr|gm|mg|ml|l|lt|cc|kg|oz|lb|fl oz|gram|kilogram|litre|liter|mililitre|milliliter|m|cm|mm|metre|meter|santimetre|centimeter|w|v|mah|kw|kcal)',
            r'(\d+(?:[\.,]\d+)?)\s*-\s*\d+(?:[\.,]\d+)?\s*(/|in|inç|inch|ekran|btu|hz|ms|devir|rpm|tb|gb|mb|cl|g|gr|gm|mg|ml|l|lt|cc|kg|oz|lb|fl oz|gram|kilogram|litre|liter|mililitre|milliliter|m|cm|mm|metre|meter|santimetre|centimeter|w|v|mah|kw|kcal)',
            r'(\d+(?:[\.,]\d+)?)\s*(/|in|inç|inch|ekran|btu|hz|ms|devir|rpm|tb|gb|mb|cl|g|gr|gm|mg|ml|l|lt|cc|kg|oz|lb|fl oz|gram|kilogram|litre|liter|mililitre|milliliter|m|cm|mm|metre|meter|santimetre|centimeter|w|v|mah|kw|kcal)'
        ]
        # ... (Senin döngü ve return mantığın aynen buraya gelecek)
        # Özetle: Senin extract_volume fonksiyonunun gövdesini buraya al.
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text)
            if match:
                if i < 2: continue
                
                if len(match.groups()) == 3 and match.group(3) is not None:
                    quantity = float(match.group(1))
                    value = float(match.group(2).replace(',', '.'))
                    unit = match.group(3)
                    value = quantity * value
                else:
                    value = float(match.group(1).replace(',', '.'))
                    unit = match.group(2)
                
                # Birim dönüşümleri
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

# --- 3. Arama Motoru (Core Logic) ---
class HybridSearchEngine:
    def __init__(self, model_name: str):
        logger.info(f"Model yükleniyor: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.bm25 = None
        self.faiss_index = None
        self.target_embeddings = None
        self.tokenized_targets = None
        self.df_target = None

    def index_targets(self, df: pd.DataFrame, text_col: str):
        """Hedef veriyi hem Dense (FAISS) hem Sparse (BM25) olarak indeksler."""
        self.df_target = df.reset_index(drop=True)
        texts = df[text_col].tolist()
        
        # 1. BM25 İndeksleme
        logger.info("BM25 indeksi oluşturuluyor...")
        self.tokenized_targets = [doc.split(" ") for doc in texts]
        self.bm25 = BM25Okapi(self.tokenized_targets)
        
        # 2. Embedding ve FAISS İndeksleme
        logger.info("Embeddingler oluşturuluyor...")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalizasyon (Cosine Similarity için kritik)
        # L2 normalize edilmiş vektörlerin Dot Product'ı = Cosine Similarity'dir.
        faiss.normalize_L2(embeddings)
        self.target_embeddings = embeddings
        
        dim = embeddings.shape[1]
        # IndexFlatIP = Inner Product (İç Çarpım). Normalize veride Cosine Sim. verir.
        self.faiss_index = faiss.IndexFlatIP(dim) 
        self.faiss_index.add(embeddings)
        logger.info(f"FAISS indeksi hazır. {self.faiss_index.ntotal} kayıt.")

    def search(self, df_source: pd.DataFrame, text_col: str, config: Config) -> pd.DataFrame:
        """
        Kaynak veriyi toplu (batch) olarak arar ve hibrit skorlar.
        """
        source_texts = df_source[text_col].tolist()
        
        # 1. Semantic Search (Toplu Arama - Batch Query)
        logger.info("Kaynak veriler encode ediliyor...")
        source_embeddings = self.model.encode(source_texts, batch_size=config.batch_size, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(source_embeddings)
        
        # Tek seferde tüm sorguları FAISS'e soruyoruz
        logger.info("Semantik arama yapılıyor...")
        D, I = self.faiss_index.search(source_embeddings, k=config.top_k)
        
        results = []
        
        # BM25 ve Merge işlemi
        logger.info("Adaylar birleştiriliyor ve skorlanıyor...")
        
        # Dış Döngü: Her bir kaynak ürün (Source Product) için döner
        for idx, row in tqdm(df_source.iterrows(), total=len(df_source)):
            source_tokens = source_texts[idx].split(" ")
            
            # FAISS Adayları
            semantic_indices = I[idx]
            semantic_scores = D[idx]
            
            # BM25 Adayları (En iyi N tanesi)
            bm25_scores = self.bm25.get_scores(source_tokens)
            best_bm25_indices = np.argsort(bm25_scores)[-config.top_k:][::-1]
            
            # Aday Birleştirme (Set ile unique yap)
            candidates = set(semantic_indices) | set(best_bm25_indices)
            
            candidates_data = []
            
            # İç Döngü: Belirlenen adaylar (Candidates) arasında döner
            for cand_idx in candidates:
                if cand_idx == -1: continue
                
                # --- SKOR HESAPLAMA BAŞLANGICI ---
                
                # 1. SBERT Skoru
                if cand_idx in semantic_indices:
                    pos = np.where(semantic_indices == cand_idx)[0][0]
                    sbert_score = semantic_scores[pos]
                else:
                    # Listede yoksa manuel hesapla
                    sbert_score = np.dot(source_embeddings[idx], self.target_embeddings[cand_idx])
                
                # 2. BM25 Skoru
                raw_bm25 = bm25_scores[cand_idx]
                norm_bm25 = 1 - (1 / (1 + raw_bm25)) 
                
                # 3. Hibrit Skor (Temel Puan)
                hybrid_score = (0.6 * sbert_score) + (0.4 * norm_bm25)
                
                # --- SENİN EKLEDİĞİN HACİM MANTIĞI BURAYA GELİYOR ---
                # ---------------------------------------------------
                
                source_volume = row.get('volume_productmain')
                # Hedef hacmi dataframe'den çekiyoruz
                target_volume = self.df_target.iloc[cand_idx].get('volume_productmatch')
                
                volume_bonus = 0
                if source_volume and target_volume:
                    # Regex işlemleri için metinleri alıyoruz
                    source_text = str(row['productmain']) if pd.notna(row['productmain']) else ""
                    target_text = str(self.df_target.iloc[cand_idx]['productmatch']) if pd.notna(self.df_target.iloc[cand_idx]['productmatch']) else ""

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
                
                # ---------------------------------------------------
                # --- HACİM MANTIĞI BİTİŞİ ---

                # 4. Final Skor (Bonus eklenmiş hali)
                final_score = min(hybrid_score + volume_bonus, 1.0)
                
                # Eşik değer kontrolü ve listeye ekleme
                # Not: Burada threshold kontrolü yapmıyoruz, hepsini alıp sıraladıktan sonra ilk 3'ü seçerken yapabiliriz 
                # veya veriyi azaltmak için burada da yapabilirsin. Şimdilik hepsini alıyoruz.
                if final_score > config.similarity_threshold:  # Veya 0.0 diyip hepsini alabilirsin
                    candidates_data.append({
                        'product': self.df_target.iloc[cand_idx]['productmatch'],
                        'code': self.df_target.iloc[cand_idx]['productcode'],
                        'score': final_score
                    })
            
            # --- SIRALAMA VE RAPORLAMA ---
            
            # Puanı en yüksekten düşüğe sırala
            candidates_data.sort(key=lambda x: x['score'], reverse=True)
            
            res_row = {'productmain': row['productmain']}
            for rank in range(3):
                suffix = rank + 1
                if rank < len(candidates_data):
                    res_row[f'match_{suffix}_product'] = candidates_data[rank]['product']
                    res_row[f'match_{suffix}_code'] = candidates_data[rank]['code']
                    res_row[f'match_{suffix}_score'] = f"%{candidates_data[rank]['score']*100:.2f}"
                else:
                    res_row[f'match_{suffix}_product'] = "N/A"
                    res_row[f'match_{suffix}_code'] = "N/A"
                    res_row[f'match_{suffix}_score'] = "N/A"
            results.append(res_row)
            
        return pd.DataFrame(results)

# --- 4. Main Execution ---
def main():
    config = Config()
    
    # Veri Yükleme (Path kontrolü yapay zeka tarafından simüle edilmiştir)
    try:
        df = pd.read_excel(config.source_path)
    except FileNotFoundError:
        logger.error("Dosya bulunamadı. Lütfen path'i Config sınıfında güncelle.")
        return

    # Ön İşleme
    logger.info("Ön işleme yapılıyor...")
    df['clean_productmain'] = df['productmain'].apply(TextProcessor.clean_text)
    df['clean_productmatch'] = df['productmatch'].apply(TextProcessor.clean_text)
    
    # Hacim çıkarma
    df['volume_productmain'] = df['productmain'].apply(TextProcessor.extract_volume)
    df['volume_productmatch'] = df['productmatch'].apply(TextProcessor.extract_volume)

    # Motoru Başlat
    engine = HybridSearchEngine(config.model_name)
    
    # Hedefleri İndeksle (Product Match column)
    # Burada varsayım: productmatch bizim veritabanımız, productmain sorgularımız.
    # Eğer aynı dosya içindeki eşleşmeyse, mantık aynıdır.
    engine.index_targets(df, 'clean_productmatch')
    
    # Arama Yap
    result_df = engine.search(df, 'clean_productmain', config)
    
    # Kayıt
    result_df.to_excel(config.output_path, index=False)
    logger.info(f"İşlem tamamlandı. Sonuçlar: {config.output_path}")

if __name__ == "__main__":
    main()