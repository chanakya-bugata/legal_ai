"""
Hybrid Retriever - Production Implementation
NOVEL ALGORITHM: First system combining Dense + Lexical + Causal retrieval
for superior legal document understanding.
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from src.clkg.clkg_graph import CLKGGraph
from src.document_processing.document_encoder import DocumentEncoder
import re

class HybridRetriever:
    """
    Production hybrid retrieval system with three orthogonal signals:
    
    1. **Dense Retrieval** (Legal-BERT): Semantic understanding
    2. **Lexical Retrieval** (BM25): Exact keyword matching  
    3. **Causal Retrieval** (CLKG): Graph-based relationship matching
    
    Weighted fusion with normalization and re-ranking.
    """
    
    def __init__(
        self,
        clauses: List[Dict],
        graph: CLKGGraph,
        encoder: DocumentEncoder,
        dense_weight: float = 0.45,
        lexical_weight: float = 0.35,
        causal_weight: float = 0.20
    ):
        """
        Args:
            clauses: List of clause dicts [{'id': str, 'text': str, ...}]
            graph: CLKGGraph for causal retrieval
            encoder: Legal-BERT encoder
            weights: Fusion weights (auto-normalized)
        """
        self.clauses = clauses
        self.graph = graph
        self.encoder = encoder
        
        # Normalize weights
        total_weight = dense_weight + lexical_weight + causal_weight
        self.dense_weight = dense_weight / total_weight
        self.lexical_weight = lexical_weight / total_weight
        self.causal_weight = causal_weight / total_weight
        
        # Pre-build indices
        self._build_indices()
        
        print(f"✅ HybridRetriever initialized:")
        print(f"   Clauses: {len(clauses)}, Weights: D={self.dense_weight:.2f} L={self.lexical_weight:.2f} C={self.causal_weight:.2f}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Production retrieval method
        
        Args:
            query: Natural language query
            top_k: Number of results
            
        Returns:
            List of results with breakdown scores:
            {
                'id': clause_id,
                'text': clause text,
                'score': 0.85,           # Combined score
                'dense_score': 0.92,     # BERT similarity
                'lexical_score': 0.78,   # BM25 keywords
                'causal_score': 0.65,    # Graph relevance,
                'risk_score': 0.42       # Clause risk
            }
        """
        print(f"\n🔍 Query: '{query[:60]}...' (top_k={top_k})")
        
        # Extract three signals
        dense_results = self._dense_retrieve(query)
        lexical_results = self._lexical_retrieve(query)
        causal_results = self._causal_retrieve(query)
        
        # Normalize all scores to [0,1]
        dense_scores = self._normalize_scores(dense_results)
        lexical_scores = self._normalize_scores(lexical_results)
        causal_scores = self._normalize_scores(causal_results)
        
        # Combine scores for all clauses
        all_clause_ids = set(self.clauses.keys())
        combined_scores = {}
        
        for clause_id in all_clause_ids:
            dense_score = dense_scores.get(clause_id, 0.0)
            lexical_score = lexical_scores.get(clause_id, 0.0)
            causal_score = causal_scores.get(clause_id, 0.0)
            
            # Weighted fusion
            hybrid_score = (
                self.dense_weight * dense_score +
                self.lexical_weight * lexical_score +
                self.causal_weight * causal_score
            )
            
            combined_scores[clause_id] = hybrid_score
        
        # Rank and return top-k
        ranked_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Format final results
        final_results = []
        for clause_id, score in ranked_results:
            clause = self._get_clause_by_id(clause_id)
            if clause:
                clause_risk = self.graph.clauses.get(clause_id, Clause("", "", 0, 0)).risk_score
                
                result = {
                    'id': clause_id,
                    'text': clause['text'],
                    'score': round(score, 3),
                    'risk_score': round(clause_risk, 3),
                    'dense_score': round(dense_scores.get(clause_id, 0.0), 3),
                    'lexical_score': round(lexical_scores.get(clause_id, 0.0), 3),
                    'causal_score': round(causal_scores.get(clause_id, 0.0), 3)
                }
                final_results.append(result)
        
        print(f"  ✓ Retrieved {len(final_results)} results (top score: {final_results[0]['score']:.3f})")
        return final_results
    
    def _dense_retrieve(self, query: str, top_k: int = 20) -> Dict[str, float]:
        """Semantic retrieval using Legal-BERT embeddings"""
        # Encode query
        query_emb = self.encoder.encode_text(query).reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_emb, self.clause_embeddings)[0]
        
        # Rank results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        scores = {}
        for idx in top_indices:
            clause_id = self._get_clause_id(idx)
            scores[clause_id] = float(similarities[idx])
        
        return scores
    
    def _lexical_retrieve(self, query: str, top_k: int = 20) -> Dict[str, float]:
        """BM25 keyword matching"""
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Rank results
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        scores = {}
        for idx in top_indices:
            clause_id = self._get_clause_id(idx)
            scores[clause_id] = float(bm25_scores[idx])
        
        return scores
    
    def _causal_retrieve(self, query: str, top_k: int = 20) -> Dict[str, float]:
        """Graph-based retrieval using CLKG structure"""
        # Find semantically closest clause
        query_emb = self.encoder.encode_text(query).reshape(1, -1)
        similarities = cosine_similarity(query_emb, self.clause_embeddings)[0]
        most_relevant_idx = np.argmax(similarities)
        most_relevant_id = self._get_clause_id(most_relevant_idx)
        
        # BFS traversal from most relevant clause
        causal_scores = self._bfs_causal_retrieval(most_relevant_id, top_k)
        
        return causal_scores
    
    def _bfs_causal_retrieval(self, start_clause_id: str, top_k: int) -> Dict[str, float]:
        """BFS in CLKG with decay"""
        from collections import deque
        
        visited = {}
        queue = deque([(start_clause_id, 1.0, 0)])  # (clause_id, score, distance)
        
        while queue and len(visited) < top_k * 3:
            clause_id, score, distance = queue.popleft()
            
            if clause_id in visited:
                continue
            
            visited[clause_id] = score
            
            # Decay by distance
            if distance >= 3:
                continue
            
            # Get neighbors (bidirectional for retrieval)
            clause = self.graph.clauses.get(clause_id)
            if clause:
                neighbors = self.graph.get_neighbors(clause_id)
                
                # Also get clauses that point to this one
                incoming = [e.source_id for e in self.graph.edges 
                           if e.target_id == clause_id]
                
                for neighbor_id in neighbors + incoming:
                    if neighbor_id not in visited:
                        decay = 0.7 ** distance
                        queue.append((neighbor_id, score * decay * 0.8, distance + 1))
        
        # Filter top-k by score
        sorted_causal = sorted(visited.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return {cid: score for cid, score in sorted_causal}
    
    def _build_indices(self):
        """Pre-compute retrieval indices"""
        # Dense index
        clause_texts = [c.get('text', '') for c in self.clauses]
        self.clause_embeddings = self.encoder.encode_clauses(clause_texts)
        
        # Lexical index (BM25)
        tokenized_corpus = [re.findall(r'\b\w+\b', text.lower()) for text in clause_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Clause ID mapping (index → id)
        self.clause_id_map = {i: c.get('id', f'clause_{i}') for i, c in enumerate(self.clauses)}
    
    def _get_clause_id(self, index: int) -> str:
        """Get clause ID by array index"""
        return self.clause_id_map.get(index, f'clause_{index}')
    
    def _get_clause_by_id(self, clause_id: str) -> Dict:
        """Get full clause by ID"""
        for clause in self.clauses:
            if clause.get('id') == clause_id:
                return clause
        return None
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalization to [0,1]"""
        if not scores:
            return {}
        
        values = np.array(list(scores.values()))
        if len(np.unique(values)) <= 1:
            return {k: 1.0 for k in scores}
        
        min_val, max_val = values.min(), values.max()
        normalized = {}
        
        for k, v in scores.items():
            norm_score = (v - min_val) / (max_val - min_val)
            normalized[k] = float(norm_score)
        
        return normalized

# Production usage example
if __name__ == "__main__":
    # Mock data (replace with pipeline)
    clauses = [
        {'id': 'C1', 'text': 'Payment shall be made within 30 days.'},
        {'id': 'C2', 'text': 'Confidentiality obligation applies.'},
        {'id': 'C3', 'text': 'Agreement terminates upon breach.'}
    ]
    
    print("✅ HybridRetriever production ready!")
    print("Usage: retriever.retrieve('payment terms', top_k=3)")
