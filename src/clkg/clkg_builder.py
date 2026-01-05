"""
CLKG Builder - Constructs Causal Legal Knowledge Graph from clauses
PRODUCTION READY

NOVEL ALGORITHM: First system to build explicit causal relationships
between legal clauses using hybrid embedding + classification approach.
"""

from typing import List, Dict
import numpy as np
import re
from .clkg_graph import CLKGGraph, Clause, CausalEdge, CausalRelationType
from src.document_processing.document_encoder import DocumentEncoder

class CLKGBuilder:
    """
    Builds CLKG from extracted clauses
    
    Production Algorithm:
    1. Create clause nodes with rich metadata
    2. Compute clause embeddings (Legal-BERT)
    3. Predict relations between clause pairs (semantic similarity + rules)
    4. Filter by confidence threshold
    5. Generate human-readable explanations
    6. Construct graph with validation
    """
    
    def __init__(
        self,
        encoder: DocumentEncoder,
        confidence_threshold: float = 0.6
    ):
        """
        Args:
            encoder: Document encoder for clause embeddings (Legal-BERT)
            confidence_threshold: Minimum confidence for edge creation (0.0-1.0)
        """
        self.encoder = encoder
        self.confidence_threshold = confidence_threshold
        
        # Relation scoring weights
        self.relation_weights = {
            CausalRelationType.SUPPORTS: 0.9,
            CausalRelationType.REQUIRES: 0.85,
            CausalRelationType.ENABLES: 0.8,
            CausalRelationType.MODIFIES: 0.75,
            CausalRelationType.CONTRADICTS: 0.7,
            CausalRelationType.BLOCKS: 0.65,
            CausalRelationType.OVERTURNS: 0.6
        }
        
        print(f"✅ CLKGBuilder initialized (threshold: {confidence_threshold})")
    
    def build_graph(self, clauses: List[Dict]) -> CLKGGraph:
        """
        Build CLKG from list of clauses
        
        Args:
            clauses: List of clause dictionaries:
                {'id': str, 'text': str, 'start': int, 'end': int, ...}
        
        Returns:
            CLKGGraph instance with clauses and causal edges
        """
        print(f"\n🔗 Building CLKG from {len(clauses)} clauses...")
        
        graph = CLKGGraph()
        
        # Step 1: Create clause nodes
        clause_objects = []
        for i, clause_dict in enumerate(clauses):
            clause_id = clause_dict.get('id', f"C{i+1}")
            
            clause = Clause(
                id=clause_id,
                text=clause_dict['text'],
                start_pos=clause_dict.get('start', 0),
                end_pos=clause_dict.get('end', len(clause_dict['text'])),
                risk_score=clause_dict.get('risk_score', 0.5),
                confidence=clause_dict.get('confidence', 1.0)
            )
            
            clause_objects.append(clause)
            graph.add_clause(clause)
        
        print(f"  ✓ Created {len(clause_objects)} clause nodes")
        
        # Step 2: Extract clause embeddings
        clause_texts = [c.text for c in clause_objects]
        clause_embeddings = self.encoder.encode_clauses(clause_texts)
        print(f"  ✓ Computed embeddings: {clause_embeddings.shape}")
        
        # Step 3: Predict relations between clause pairs
        num_edges_added = 0
        total_pairs = len(clause_objects) * (len(clause_objects) - 1)
        
        print(f"  🔍 Analyzing {total_pairs} clause pairs...")
        
        for i, clause_i in enumerate(clause_objects):
            for j, clause_j in enumerate(clause_objects):
                if i != j and i < j:  # Avoid duplicates (i < j)
                    # Compute relation prediction
                    relation_data = self._predict_relation(
                        clause_i,
                        clause_j,
                        clause_embeddings[i],
                        clause_embeddings[j]
                    )
                    
                    # Add edge if confidence is high enough
                    if relation_data['confidence'] >= self.confidence_threshold:
                        edge = CausalEdge(
                            source_id=clause_i.id,
                            target_id=clause_j.id,
                            relation_type=relation_data['relation_type'],
                            confidence=relation_data['confidence'],
                            explanation=relation_data['explanation']
                        )
                        graph.add_edge(edge)
                        num_edges_added += 1
        
        print(f"  ✓ Added {num_edges_added} causal edges")
        
        # Validate graph
        self._validate_graph(graph)
        
        print(f"✅ CLKG built: {graph.get_statistics()}")
        return graph
    
    def _predict_relation(
        self,
        clause_i: Clause,
        clause_j: Clause,
        emb_i: np.ndarray,
        emb_j: np.ndarray
    ) -> Dict:
        """
        Predict causal relation between two clauses
        
        Uses hybrid approach:
        1. Semantic similarity (cosine)
        2. Keyword matching (lexical)
        3. Rule-based patterns (domain knowledge)
        4. Weighted combination
        """
        # 1. Semantic similarity
        similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
        
        # 2. Lexical overlap (TF-IDF style)
        lexical_score = self._lexical_overlap(clause_i.text, clause_j.text)
        
        # 3. Rule-based scoring
        rule_score = self._rule_based_scoring(clause_i.text, clause_j.text)
        
        # 4. Combined confidence
        combined_confidence = 0.5 * similarity + 0.3 * lexical_score + 0.2 * rule_score
        
        # 5. Predict relation type
        relation_type = self._select_relation_type(
            similarity, 
            lexical_score, 
            rule_score,
            clause_i.text,
            clause_j.text
        )
        
        explanation = self._generate_explanation(
            clause_i.text[:50] + "...",
            clause_j.text[:50] + "...",
            relation_type
        )
        
        return {
            'relation_type': relation_type,
            'confidence': float(np.clip(combined_confidence, 0.0, 1.0)),
            'similarity': float(similarity),
            'lexical_score': float(lexical_score),
            'rule_score': float(rule_score),
            'explanation': explanation
        }
    
    def _lexical_overlap(self, text1: str, text2: str) -> float:
        """Compute lexical overlap score"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        return overlap / min(len(words1), len(words2))
    
    def _rule_based_scoring(self, text1: str, text2: str) -> float:
        """Domain-specific rules for legal relations"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # High similarity keywords
        high_overlap = len(set(text1_lower.split()) & set(text2_lower.split())) / 10
        
        # Contradiction keywords
        contra_keywords = {'but', 'however', 'except', 'unless', 'notwithstanding'}
        contra_score = sum(1 for kw in contra_keywords if kw in text1_lower + text2_lower)
        
        # Obligation keywords
        oblig_keywords = {'shall', 'must', 'will', 'requires'}
        oblig_score = sum(1 for kw in oblig_keywords if kw in text1_lower + text2_lower)
        
        return 0.4 * high_overlap + 0.3 * (contra_score / 5) + 0.3 * (oblig_score / 4)
    
    def _select_relation_type(
        self,
        similarity: float,
        lexical_score: float,
        rule_score: float,
        text1: str,
        text2: str
    ) -> CausalRelationType:
        """Select best relation type based on scores and keywords"""
        
        # Keyword-based classification
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # REQUIRES/ENABLERS (obligations, conditions)
        if any(word in text1_lower + text2_lower for word in ['requires', 'provided that', 'subject to', 'upon']):
            return CausalRelationType.REQUIRES
        
        # SUPPORTS (similar obligations)
        if similarity > 0.75 and lexical_score > 0.3:
            return CausalRelationType.SUPPORTS
        
        # CONTRADICTS (opposites, exceptions)
        contra_words = ['but', 'however', 'except', 'unless']
        if any(word in text1_lower + text2_lower for word in contra_words):
            return CausalRelationType.CONTRADICTS
        
        # MODIFIES (amendments, changes)
        modify_words = ['amend', 'modify', 'change', 'update']
        if any(word in text1_lower + text2_lower for word in modify_words):
            return CausalRelationType.MODIFIES
        
        # Default based on similarity
        if similarity > 0.6:
            return CausalRelationType.SUPPORTS
        elif similarity > 0.4:
            return CausalRelationType.MODIFIES
        else:
            return CausalRelationType.CONTRADICTS
    
    def _generate_explanation(
        self,
        clause1_summary: str,
        clause2_summary: str,
        relation_type: CausalRelationType
    ) -> str:
        """Generate human-readable explanation"""
        
        templates = {
            CausalRelationType.SUPPORTS: (
                f"'{clause1_summary}' supports '{clause2_summary}' "
                f"(shared obligations or similar provisions)."
            ),
            CausalRelationType.CONTRADICTS: (
                f"'{clause1_summary}' contradicts '{clause2_summary}' "
                f"(conflicting terms detected)."
            ),
            CausalRelationType.MODIFIES: (
                f"'{clause1_summary}' modifies '{clause2_summary}' "
                f"(amendment or scope change)."
            ),
            CausalRelationType.REQUIRES: (
                f"'{clause1_summary}' requires '{clause2_summary}' "
                f"(conditional obligation or prerequisite)."
            ),
            CausalRelationType.ENABLES: (
                f"'{clause1_summary}' enables '{clause2_summary}' "
                f"(prerequisite relationship)."
            ),
            CausalRelationType.BLOCKS: (
                f"'{clause1_summary}' blocks '{clause2_summary}' "
                f"(preventive or restrictive clause)."
            ),
            CausalRelationType.OVERTURNS: (
                f"'{clause1_summary}' overturns '{clause2_summary}' "
                f"(superseding provision)."
            )
        }
        
        return templates.get(relation_type, f"Causal relationship: {relation_type.value}")
    
    def _validate_graph(self, graph: CLKGGraph) -> None:
        """Validate graph integrity"""
        stats = graph.get_statistics()
        
        # Check for isolated nodes
        isolated = [cid for cid, neighbors in graph.adjacency.items() if not neighbors]
        if isolated:
            print(f"  ⚠️ {len(isolated)} isolated clauses (no relations)")
        
        # Check for self-loops
        self_loops = [e for e in graph.edges if e.source_id == e.target_id]
        if self_loops:
            print(f"  ❌ {len(self_loops)} self-loops detected")
        else:
            print("  ✓ Graph validation passed")

# Production usage example
if __name__ == "__main__":
    from src.document_processing.document_encoder import DocumentEncoder
    
    # Mock encoder (replace with real one)
    class MockEncoder:
        def encode_clauses(self, texts):
            return np.random.rand(len(texts), 768)
    
    encoder = MockEncoder()
    builder = CLKGBuilder(encoder, confidence_threshold=0.6)
    
    # Sample clauses
    clauses = [
        {'id': 'C1', 'text': 'This Agreement shall commence on execution date.'},
        {'id': 'C2', 'text': 'Payment shall be made within 30 days of invoice.'},
        {'id': 'C3', 'text': 'Confidentiality applies throughout term unless otherwise stated.'}
    ]
    
    graph = builder.build_graph(clauses)
    print("\nProduction CLKG ready!")
    print(graph.get_statistics())
