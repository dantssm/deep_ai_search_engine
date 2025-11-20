import hashlib
import httpx
import re
from typing import List, Dict, Optional, Set
from collections import Counter
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.logger import log_rag


class RAGStore:
    """RAG store wrapper around ChromaDB with custom hybrid scoring logic"""
    
    JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
    JINA_MODEL = "jina-embeddings-v2-base-en"
    
    def __init__(self, jina_key: str):
        """
        Initialize the RAG store.

        Args:
            jina_key (str): API key for Jina Embeddings.
        """
        self.jina_key = jina_key
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        self._collections: Dict[str, chromadb.Collection] = {}
        self._chunk_metadata: Dict[str, List[Dict]] = {}
    
    def _get_collection(self, collection_id: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection."""
        if collection_id not in self._collections:
            self._collections[collection_id] = self.client.get_or_create_collection(
                name=collection_id, metadata={"hnsw:space": "cosine"}
            )
        return self._collections[collection_id]
    
    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Jina API.
        
        Args:
            texts (List[str]): List of strings to embed.
            
        Returns:
            List[List[float]]: List of vector embeddings.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.JINA_EMBED_URL,
                headers={"Authorization": f"Bearer {self.jina_key}"},
                json={"input": texts, "model": self.JINA_MODEL}
            )
            resp.raise_for_status()
            return [d["embedding"] for d in resp.json()["data"]]
    
    def _context_aware_chunk(self, text: str, url: str, max_size: int = 800) -> List[Dict]:
        """
        Split text into chunks while preserving headers.

        Use of RecursiveCharacterTextSplitter and attach the most recent markdown header (# Header) 
        to each chunk so the LLM knows what the text is about.

        Args:
            text (str): Raw text content.
            url (str): Source URL.
            max_size (int): Max characters per chunk.

        Returns:
            List[Dict]: List of chunk objects.
        """
        text = text.strip()
        if not text: return []
        
        headers = re.findall(r'^#+\s+(.+)$|<h[1-6]>(.+?)</h[1-6]>', text, re.MULTILINE)
        current_header = ""
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        chunks_text = splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            for h_match in headers:
                h = h_match[0] or h_match[1]
                if h in chunk_text:
                    current_header = h
                    break
            
            chunks.append({
                "text": chunk_text.strip(),
                "header": current_header,
                "url": url,
                "chunk_idx": i
            })
        
        return [c for c in chunks if len(c["text"]) > 50]
    
    async def add_documents(self, collection_id: str, docs: List[Dict], quality_scores: Optional[Dict[str, float]] = None) -> int:
        """
        Process and index documents into vector store.

        Args:
            collection_id (str): Unique ID for this research session.
            docs (List[Dict]): List of docs with 'url' and 'content'.
            quality_scores (Optional[Dict]): Map of URL -> reliability score (0.0 - 1.0).

        Returns:
            int: Number of chunks successfully added.
        """
        collection = self._get_collection(collection_id)
        
        all_chunks, all_metas, all_ids = [], [], []
        
        for doc in docs:
            url = doc.get("url", "")
            chunks = self._context_aware_chunk(doc.get("content", ""), url)
            
            title = doc.get("title", "")
            quality = quality_scores.get(url, 0.8) if quality_scores else 0.8
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk["text"])
                all_metas.append({
                    "url": url,
                    "title": title,
                    "chunk_idx": chunk.get("chunk_idx", i),
                    "header": chunk.get("header", ""),
                    "quality": quality
                })
                all_ids.append(hashlib.md5(f"{url}_{i}".encode()).hexdigest())
        
        if not all_chunks:
            return 0
        
        embeddings = await self._embed(all_chunks)
        collection.add(ids=all_ids, embeddings=embeddings, documents=all_chunks, metadatas=all_metas)
        
        if collection_id not in self._chunk_metadata:
            self._chunk_metadata[collection_id] = []
        self._chunk_metadata[collection_id].extend(all_metas)
        
        log_rag(f"Added {len(all_chunks)} chunks to {collection_id}")
        return len(all_chunks)
    
    def _hybrid_score(self, chunk: Dict, query: str, semantic_weight: float = 0.7) -> float:
        """
        Combine Vector Similarity (Semantic) with Keyword Match (BM25-lite).
        
        Args:
            chunk (Dict): The chunk data including semantic score.
            query (str): The search query.
            semantic_weight (float): How much to trust the vector score vs keyword score.

        Returns:
            float: A unified score between 0.0 and 1.0.
        """
        semantic = chunk.get("score", 0)
        
        content_lower = chunk["content"].lower()
        query_terms = query.lower().split()
        content_words = content_lower.split()
        term_freqs = Counter(content_words)
        
        k1 = 1.5 # saturation parameter
        b = 0.75 # length penalty parameter
        avg_doc_len = 100
        doc_len = len(content_words)
        
        bm25_score = 0.0
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            if tf > 0:
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                bm25_score += (numerator / denominator)
        
        bm25_normalized = min(bm25_score / max(len(query_terms), 1), 1.0) if query_terms else 0.0
        
        keyword_weight = 1.0 - semantic_weight
        return (semantic_weight * semantic) + (keyword_weight * bm25_normalized)
    
    async def search(
        self, collection_id: str, query: str, n: int = 5, diversity_weight: float = 0.3, use_mmr: bool = True
    ) -> List[Dict]:
        """
        Main search function executing the full retrieval pipeline.

        1. Vector Search (Semantic)
        2. Hybrid Scoring (Semantic + Keyword)
        3. Quality Weighting (Source Reliability)
        4. MMR Re-ranking (Diversity)

        Args:
            collection_id (str): ID of the collection to search.
            query (str): The user's query.
            n (int): Number of results to return.
            diversity_weight (float): How much to penalize duplicate sources.
            use_mmr (bool): Whether to use Maximal Marginal Relevance.

        Returns:
            List[Dict]: Top N most relevant and diverse chunks.
        """
        collection = self._get_collection(collection_id)
        
        if collection.count() == 0:
            return []
        
        all_chunks = []
        
        query_emb = await self._embed([query])
        
        results = collection.query(query_embeddings=query_emb,n_results=min(n * 3, collection.count()) )
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            
            chunk = {
                "content": results["documents"][0][i],
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
                "header": meta.get("header", ""),
                "quality": meta.get("quality", 0.8),
                "score": 1 - results["distances"][0][i] if results.get("distances") else 0
            }
            
            chunk["score"] = self._hybrid_score(chunk, query)
            chunk["score"] *= chunk["quality"]
            
            all_chunks.append(chunk)
    
        if use_mmr and len(all_chunks) > 1:
            mmr_scores = self._calculate_mmr_scores(all_chunks)
            for i, chunk in enumerate(all_chunks):
                chunk["mmr_score"] = mmr_scores[i]
            
            all_chunks.sort(key=lambda c: c["mmr_score"], reverse=True)
        else:
            all_chunks.sort(key=lambda c: c["score"], reverse=True)
        
        if diversity_weight > 0:
            all_chunks = self._apply_diversity_boost(all_chunks, diversity_weight)
        
        final_chunks = all_chunks[:n]
        log_rag(f"Retrieved {len(final_chunks)} chunks from {collection_id}")
        return final_chunks

    def _calculate_mmr_scores(self, chunks: List[Dict], lambda_param: float = 0.7) -> List[float]:
        """
        Calculate Maximal Marginal Relevance.

        Args:
            chunks (List[Dict]): List of chunk dicts with 'content' and 'score'.
            lambda_param (float): Trade-off parameter. 1.0 = Pure Relevance, 0.0 = Pure Diversity.

        Returns:
            List[float]: MMR scores for each chunk.
        """
        if not chunks: return []
        mmr_scores = []
        selected_indices: Set[int] = set()
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                mmr_scores.append(chunk.get("score", 0))
                selected_indices.add(i)
                continue

            max_sim = 0.0
            for j in selected_indices:
                words_i = set(chunk["content"].lower().split())
                words_j = set(chunks[j]["content"].lower().split())
                if not words_i or not words_j:
                    sim = 0.0
                else:
                    sim = len(words_i & words_j) / len(words_i | words_j)
                max_sim = max(max_sim, sim)
            
            # Relevance - (1-lambda) * Redundancy
            mmr = (lambda_param * chunk.get("score", 0)) - ((1 - lambda_param) * max_sim)
            mmr_scores.append(mmr)
            selected_indices.add(i)
        
        return mmr_scores
    
    def _apply_diversity_boost(self, chunks: List[Dict], weight: float) -> List[Dict]:
        """
        Reduce score if URL has appeared frequently.

        Args:
            chunks (List[Dict]): List of chunk dicts with 'url' and 'score'.
            weight (float): How much to penalize duplicate sources.

        Returns:
            List[Dict]: Re-ranked chunks with diversity applied.
        """
        url_counts = Counter(c["url"] for c in chunks)
        
        for chunk in chunks:
            count = url_counts[chunk["url"]]
            diversity_factor = 1.0 / count 
            
            base_score = chunk.get("mmr_score", chunk.get("score", 0))
            chunk["final_score"] = ((1 - weight) * base_score) + (weight * diversity_factor)
        
        chunks.sort(key=lambda c: c["final_score"], reverse=True)
        return chunks

    def clear(self, collection_id: str):
        """Delete a specific collection."""
        if collection_id in self._collections:
            try:
                self.client.delete_collection(collection_id)
            except Exception:
                pass
            del self._collections[collection_id]
        
        if collection_id in self._chunk_metadata:
            del self._chunk_metadata[collection_id]
            
    def clear_all(self):
        """Delete everything (Session cleanup)."""
        import gc
        for cid in list(self._collections.keys()):
            self.clear(cid)
        try:
            self.client.reset()
        except:
            pass
        self._chunk_metadata.clear()
        gc.collect()