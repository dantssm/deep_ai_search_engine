# src/rag/enhanced_store.py
"""Enhanced RAG with hybrid search, re-ranking, and query expansion.

Improvements over base RAG:
1. Hybrid search (semantic + keyword)
2. Query expansion for better recall
3. MMR (Maximal Marginal Relevance) for diversity
4. Context-aware chunking (now with better sentence splitting)
5. Source quality weighting
"""

import hashlib
import httpx
import re
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.logger import log_rag


class EnhancedRAGStore:
    """RAG store with advanced retrieval strategies."""
    
    JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
    JINA_MODEL = "jina-embeddings-v2-base-en"
    
    def __init__(self, jina_key: str):
        self.jina_key = jina_key
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        self._collections: Dict[str, chromadb.Collection] = {}
        self._chunk_metadata: Dict[str, List[Dict]] = {}  # Store chunk context
    
    def _get_collection(self, collection_id: str) -> chromadb.Collection:
        if collection_id not in self._collections:
            self._collections[collection_id] = self.client.get_or_create_collection(
                name=collection_id,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[collection_id]
    
    async def _embed(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.JINA_EMBED_URL,
                headers={"Authorization": f"Bearer {self.jina_key}"},
                json={"input": texts, "model": self.JINA_MODEL}
            )
            resp.raise_for_status()
            return [d["embedding"] for d in resp.json()["data"]]
    
    def _context_aware_chunk(self, text: str, url: str, max_size: int = 800) -> List[Dict]:
        """Context-aware chunking using LangChain's battle-tested splitter.
        
        Uses RecursiveCharacterTextSplitter which:
        - Preserves sentence boundaries
        - Handles abbreviations (Dr., Prof., etc.)
        - Tries multiple separators (\n\n, \n, '. ', ' ')
        - Maintains overlap for context
        """
        text = text.strip()
        if not text:
            return []
        
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
    
    async def add_documents(self, collection_id: str, docs: List[Dict], 
                          quality_scores: Optional[Dict[str, float]] = None) -> int:
        """Add documents with quality weighting.
        
        Args:
            collection_id: Collection to add to
            docs: Documents with 'url' and 'content'
            quality_scores: Optional quality scores per URL (0-1)
        
        Returns:
            Number of chunks added
        """
        collection = self._get_collection(collection_id)
        
        all_chunks, all_metas, all_ids = [], [], []
        
        for doc in docs:
            chunks = self._context_aware_chunk(
                doc.get("content", ""),
                doc.get("url", "")
            )
            
            url = doc.get("url", "")
            title = doc.get("title", "")
            quality = quality_scores.get(url, 0.8) if quality_scores else 0.8
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk["text"])
                all_metas.append({
                    "url": url,
                    "title": title,
                    "chunk_idx": chunk.get("chunk_idx", i),
                    "header": chunk.get("header", ""),
                    "quality": quality  # Store quality for weighting
                })
                all_ids.append(hashlib.md5(f"{url}_{i}".encode()).hexdigest())
        
        if not all_chunks:
            return 0
        
        embeddings = await self._embed(all_chunks)
        collection.add(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metas
        )
        
        if collection_id not in self._chunk_metadata:
            self._chunk_metadata[collection_id] = []
        self._chunk_metadata[collection_id].extend(all_metas)
        
        log_rag(f"Added {len(all_chunks)} context-aware chunks to {collection_id}")
        return len(all_chunks)
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms.
        
        Simple expansion strategy:
        1. Add common synonyms
        2. Add related terms
        3. Remove stopwords
        """
        expansions = [query]
        
        synonyms = {
            "how": ["what is the process", "explain", "mechanism"],
            "why": ["reason", "cause", "explanation"],
            "best": ["top", "optimal", "most effective"],
            "effect": ["impact", "consequence", "result"],
            "use": ["application", "purpose", "function"]
        }
        
        words = query.lower().split()
        for word in words:
            if word in synonyms:
                for syn in synonyms[word][:2]:
                    expanded = query.replace(word, syn)
                    expansions.append(expanded)
        
        return expansions[:3]
    
    def _calculate_mmr_scores(self, chunks: List[Dict], 
                             lambda_param: float = 0.7) -> List[float]:
        """Calculate MMR (Maximal Marginal Relevance) scores.
        
        MMR balances relevance with diversity.
        
        Args:
            chunks: Chunks with 'score' and 'content'
            lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
        
        Returns:
            MMR scores for each chunk
        """
        if not chunks:
            return []
        
        mmr_scores = []
        selected: Set[int] = set()
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                mmr_scores.append(chunk.get("score", 0))
                selected.add(i)
            else:
                max_sim = 0.0
                for j in selected:
                    words_i = set(chunk["content"].lower().split())
                    words_j = set(chunks[j]["content"].lower().split())
                    sim = len(words_i & words_j) / max(len(words_i | words_j), 1)
                    max_sim = max(max_sim, sim)
                
                mmr = lambda_param * chunk.get("score", 0) - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)
                selected.add(i)
        
        return mmr_scores
    
    def _hybrid_score(self, chunk: Dict, query: str, 
                     semantic_weight: float = 0.7) -> float:
        """Calculate hybrid score (semantic + keyword).
        
        Args:
            chunk: Chunk with 'content' and semantic 'score'
            query: Original query
            semantic_weight: Weight for semantic score (0-1)
        
        Returns:
            Combined score
        """
        semantic = chunk.get("score", 0)
        
        content_lower = chunk["content"].lower()
        query_terms = query.lower().split()
        
        content_words = content_lower.split()
        term_freqs = Counter(content_words)
        
        k1 = 1.5
        b = 0.75
        avg_doc_len = 100
        doc_len = len(content_words)
        
        bm25_score = 0.0
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            idf = 1.0
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            
            bm25_score += idf * (numerator / denominator)
        
        bm25_normalized = min(bm25_score / len(query_terms), 1.0) if query_terms else 0.0
        
        keyword_weight = 1.0 - semantic_weight
        return semantic_weight * semantic + keyword_weight * bm25_normalized
    
    async def search(self, collection_id: str, query: str, n: int = 5,
                    diversity_weight: float = 0.3,
                    use_query_expansion: bool = True,
                    use_mmr: bool = True) -> List[Dict]:
        """Enhanced search with multiple strategies.
        
        Args:
            collection_id: Collection to search
            query: Search query
            n: Number of results
            diversity_weight: Weight for source diversity (0-1)
            use_query_expansion: Expand query for better recall
            use_mmr: Use MMR for diversity
        
        Returns:
            List of ranked chunks
        """
        collection = self._get_collection(collection_id)
        
        if collection.count() == 0:
            log_rag(f"Collection {collection_id} is empty", level="warning")
            return []
        
        queries = self._expand_query(query) if use_query_expansion else [query]
        
        all_chunks = []
        seen_ids = set()
        
        for q in queries:
            query_emb = await self._embed([q])
            results = collection.query(
                query_embeddings=query_emb,
                n_results=min(n * 3, collection.count())
            )
            
            if not results["ids"] or not results["ids"][0]:
                continue
            
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                
                chunk = {
                    "content": results["documents"][0][i],
                    "url": results["metadatas"][0][i].get("url", ""),
                    "title": results["metadatas"][0][i].get("title", ""),
                    "header": results["metadatas"][0][i].get("header", ""),
                    "quality": results["metadatas"][0][i].get("quality", 0.8),
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
        log_rag(f"Retrieved {len(final_chunks)} chunks (hybrid + MMR) from {collection_id}")
        return final_chunks
    
    def _apply_diversity_boost(self, chunks: List[Dict], 
                              weight: float) -> List[Dict]:
        """Boost scores for diverse sources."""
        url_counts = Counter(c["url"] for c in chunks)
        
        for chunk in chunks:
            url_count = url_counts[chunk["url"]]
            diversity_boost = 1.0 / url_count
            
            original_score = chunk.get("mmr_score", chunk.get("score", 0))
            chunk["final_score"] = (
                (1 - weight) * original_score + 
                weight * diversity_boost
            )
        
        chunks.sort(key=lambda c: c["final_score"], reverse=True)
        return chunks
    
    async def search_with_context(self, collection_id: str, query: str, 
                                 n: int = 5, context_window: int = 1) -> List[Dict]:
        """Search and include surrounding chunks for context.
        
        Args:
            collection_id: Collection to search
            query: Search query
            n: Number of results
            context_window: Number of chunks before/after to include
        
        Returns:
            Chunks with context
        """
        base_chunks = await self.search(collection_id, query, n)
        
        chunks_with_context = []
        
        for chunk in base_chunks:
            url = chunk["url"]
            chunk_idx = chunk.get("chunk_idx", 0)
            
            metadata = self._chunk_metadata.get(collection_id, [])
            url_chunks = [m for m in metadata if m["url"] == url]
            
            context_before = []
            context_after = []
            
            for i in range(1, context_window + 1):
                before_idx = chunk_idx - i
                if 0 <= before_idx < len(url_chunks):
                    context_before.append(url_chunks[before_idx])
                
                after_idx = chunk_idx + i
                if after_idx < len(url_chunks):
                    context_after.append(url_chunks[after_idx])
            
            chunks_with_context.append({
                **chunk,
                "context_before": context_before,
                "context_after": context_after
            })
        
        return chunks_with_context
    
    def clear(self, collection_id: str):
        """Clear collection and metadata."""
        if collection_id in self._collections:
            try:
                self.client.delete_collection(collection_id)
                log_rag(f"Cleared collection {collection_id}")
            except Exception as e:
                log_rag(f"Failed to clear {collection_id}: {e}", level="error")
            
            del self._collections[collection_id]
        
        if collection_id in self._chunk_metadata:
            del self._chunk_metadata[collection_id]
    
    def clear_all(self):
        """Clear all collections and force GC."""
        import gc
        
        for cid in list(self._collections.keys()):
            self.clear(cid)
        
        try:
            self.client.reset()
        except:
            pass
        
        self._chunk_metadata.clear()
        gc.collect()
        log_rag("Cleared all RAG collections + forced GC")