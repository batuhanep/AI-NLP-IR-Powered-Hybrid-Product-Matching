import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging
import sys

# 1. LOGGING AYARLARI (Gözlemlebilirlik)
# Kodun ne yaptığını terminalden profesyonelce takip etmek için.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class HybridProductMatcher:
    """
    BM25 (Seyrek) ve Embedding (Yoğun) aramayı birleştiren,
    ardından Cross-Encoder ile en iyi adayları seçen modüler sınıf.
    """
    
    def __init__(self, embedding_model_name: str = "BAAI/bge-m3", 
                 reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
                 use_gpu: bool = True):
        
        #self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available(): # Mac M1/M2/M3 Desteği
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"
        logger.info(f"Cihaz kullanılıyor: {self.device}")

        # Embedding Modelini Yükle
        logger.info(f"Embedding modeli yükleniyor: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        
        # Cross-Encoder (Reranker) Yükle
        logger.info(f"Reranker modeli yükleniyor: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)
        
        # Veri yapıları
        self.corpus_df = None
        self.bm25 = None
        self.faiss_index = None
        self.corpus_embeddings = None
        
    def _preprocess(self, text: str) -> str:
        """Basit metin temizliği. Küçük harf, boşluk düzenleme."""
        if not isinstance(text, str):
            return ""
        return " ".join(text.lower().split())

    def _tokenize(self, text: str) -> List[str]:
        """BM25 için basit tokenization."""
        return self._preprocess(text).split()

    def fit(self, df_corpus: pd.DataFrame, text_column: str, code_column: str):
        """
        Arama yapılacak havuzu (Product Match listesi) indeksler.
        
        Args:
            df_corpus: Arama havuzu DataFrame'i
            text_column: Ürün isimlerinin olduğu sütun adı
            code_column: Ürün url/kod sütun adı
        """
        logger.info("Veri havuzu (Corpus) indeksleniyor...")
        
        # Veriyi temizle ve sakla
        self.corpus_df = df_corpus.copy()
        self.corpus_df[text_column] = self.corpus_df[text_column].fillna("")
        self.corpus_texts = self.corpus_df[text_column].tolist()
        
        # 1. BM25 İndeksi Oluştur
        logger.info("BM25 indeksi oluşturuluyor...")
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. FAISS (Vektör) İndeksi Oluştur
        logger.info("Vektör embeddingleri oluşturuluyor (Bu işlem veri boyutuna göre zaman alabilir)...")
        # Batching: Bellek taşmasını önlemek için parça parça encode et
        self.corpus_embeddings = self.embedder.encode(
            self.corpus_texts, 
            batch_size=32, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True # Cosine similarity için normalizasyon şart
        )
        
        # FAISS Index Flat IP (Inner Product) -> Normalize edilmiş vektörlerde Cosine Similarity'e eşittir.
        dimension = self.corpus_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # GPU varsa FAISS'i GPU'ya taşı (Opsiyonel, büyük veride hız katar)
        if self.device == "cuda":
            try:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
            except:
                logger.warning("FAISS GPU'ya taşınamadı, CPU devam ediliyor.")
                
        self.faiss_index.add(self.corpus_embeddings)
        logger.info(f"İndeksleme tamamlandı. Toplam ürün sayısı: {len(self.corpus_texts)}")

    def search(self, query: str, top_k_candidates: int = 20, top_n_final: int = 3) -> List[Dict]:
        """
        Tek bir sorgu için Hybrid Search ve Reranking yapar.
        """
        query_clean = self._preprocess(query)
        
        # --- ADIM 1: Aday Belirleme (Candidate Generation) ---
        
        # A) BM25 ile Adaylar
        tokenized_query = self._tokenize(query_clean)
        # BM25 skorlarını al, en yüksek k tanesinin indeksini bul
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k_candidates]
        
        # B) FAISS (BGE-M3) ile Adaylar
        query_embedding = self.embedder.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        # D, I = Distances, Indices
        _, faiss_top_indices = self.faiss_index.search(query_embedding, top_k_candidates)
        faiss_top_indices = faiss_top_indices[0] # Tek sorgu olduğu için 0. index
        
        # C) Adayları Birleştir (Union - Tekrarları önle)
        candidate_indices = list(set(bm25_top_indices) | set(faiss_top_indices))
        
        # --- ADIM 2: Cross-Encoder ile Sıralama (Reranking) ---
        
        # Cross-Encoder için (Query, Document) çiftleri hazırla
        candidates = []
        for idx in candidate_indices:
            candidates.append([query, self.corpus_texts[idx]])
            
        if not candidates:
            return []
            
        # Skorla
        rerank_scores = self.reranker.predict(candidates)
        
        # Skorlara göre sırala
        # results listesi: (skor, orijinal_index)
        results_with_idx = sorted(
            list(zip(rerank_scores, candidate_indices)), 
            key=lambda x: x[0], 
            reverse=True
        )[:top_n_final]
        
        # Çıktıyı hazırla
        final_output = []
        for score, idx in results_with_idx:
            row_data = self.corpus_df.iloc[idx]
            final_output.append({
                "match_score": float(score),
                "matched_product": row_data["productmatch"], # 'text_column'
                "matched_code": row_data["productcode"]      # 'code_column'
            })
            
        return final_output
    
    def _sigmoid(self, x):
        """
        Logit skorunu (örn: 4.5 veya -2.1) 0 ile 1 arasında olasılığa çevirir.
        Cross-Encoder çıktılarını yüzdesel göstermek için gereklidir.
        """
        return 1 / (1 + np.exp(-x))

    def batch_process(self, query_list: List[str], output_file: str):
        """
        Toplu işlem yapar. 
        ÖNEMLİ: Input listesindeki satır sırasını ve sayısını %100 korur.
        """
        logger.info(f"Toplam {len(query_list)} satır işlenecek (Sıra korunarak)...")
        
        all_results = []
        
        # TQDM barı ile ilerlemeyi göster
        for query in tqdm(query_list, desc="Eşleştiriliyor"):
            
            row = {"productmain": query} # Orijinal sorguyu yaz
            
            # Veri Validasyonu: Eğer hücre boşsa veya string değilse (NaN), işlemi pas geç ama satırı koru.
            if not isinstance(query, str) or not query.strip():
                # Boş satır için boş sütunlar ekle
                for i in range(1, 4): # Top 3
                    row[f"match_{i}_product"] = None
                    row[f"match_{i}_code"] = None
                    row[f"match_{i}_score"] = None
            else:
                # Dolu satır ise Arama Yap
                matches = self.search(query)
                
                for i, match in enumerate(matches):
                    rank = i + 1
                    row[f"match_{rank}_product"] = match["matched_product"]
                    row[f"match_{rank}_code"] = match["matched_code"]
                    
                    # SKOR DÖNÜŞÜMÜ (Logit -> Yüzde)
                    # Model ham skor verir, Sigmoid ile % formatına çeviriyoruz.
                    raw_score = match["match_score"]
                    percentage_score = self._sigmoid(raw_score) * 100
                    
                    # Excel'de güzel görünmesi için string formatlama: "%98.50"
                    row[f"match_{rank}_score"] = f"%{percentage_score:.2f}"
            
            all_results.append(row)
            
        # DataFrame oluştur ve kaydet
        df_results = pd.DataFrame(all_results)
        df_results.to_excel(output_file, index=False)
        logger.info(f"Sonuçlar başarıyla kaydedildi: {output_file}")
            

# --- KULLANIM SENARYOSU (MAIN) ---

if __name__ == "__main__":
    
    FILE_PATH = "/Users/batuhanozdemir/Desktop/datascience-ops-software/Datascience/ninja1.xlsx" 
    OUTPUT_PATH = "eslesmis_urunlerninja1.xlsx"
    
    try:
        # Veriyi oku
        df = pd.read_excel(FILE_PATH)
        
        # 1. Katalog Verisini Hazırla (Burası aynı kalabilir, arama havuzu temiz olmalı)
        catalog_df = df[["productmatch", "productcode"]].dropna(subset=["productmatch"]).drop_duplicates(subset=["productmatch"]).reset_index(drop=True)
        
        # 2. Sorgu Verisini Hazırla - DEĞİŞİKLİK BURADA
        # .dropna() veya .unique() YAPMIYORUZ.
        # Böylece Excel'deki 5. satır neyse, çıktıda da 5. satır o olacak.
        queries = df["productmain"].tolist()
        
        if len(catalog_df) == 0:
            logger.error("Aranacak ürün havuzu (productmatch) boş!")
        else:
            matcher = HybridProductMatcher()
            
            # Modeli Eğit
            matcher.fit(catalog_df, text_column="productmatch", code_column="productcode")
            
            # Eşleştirmeyi Başlat
            matcher.batch_process(queries, OUTPUT_PATH)
            
    except Exception as e:
        logger.error(f"Hata oluştu: {e}")