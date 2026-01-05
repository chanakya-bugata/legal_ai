"""
Causal Legal Knowledge Graph (CLKG) Data Structure - PRODUCTION READY

NOVEL CONTRIBUTION: First system to model explicit causal relationships
between legal clauses (not just similarity).
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
import networkx as nx
import numpy as np
from typing import Union

class CausalRelationType(Enum):
    """Causal relation types between clauses"""
    SUPPORTS = "SUPPORTS"          # A enables fulfillment of B
    CONTRADICTS = "CONTRADICTS"    # A conflicts with B
    MODIFIES = "MODIFIES"          # A changes scope of B
    OVERTURNS = "OVERTURNS"        # A voids/replaces B
    ENABLES = "ENABLES"            # A is prerequisite for B
    BLOCKS = "BLOCKS"              # A prevents B
    REQUIRES = "REQUIRES"          # B mandatory if A occurs

@dataclass
class Clause:
    """Represents a legal clause"""
    id: str
    text: str
    start_pos: int
    end_pos: int
    entities: List[Dict] = None     # Parties, amounts, dates
    obligations: List[Dict] = None  # What each party must do
    conditions: List[Dict] = None   # When obligations apply
    risk_score: float = 0.0         # Risk score (0-1)
    confidence: float = 1.0         # Extraction confidence
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.obligations is None:
            self.obligations = []
        if self.conditions is None:
            self.conditions = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'text': self.text,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'entities': self.entities,
            'obligations': self.obligations,
            'conditions': self.conditions,
            'risk_score': self.risk_score,
            'confidence': self.confidence
        }

@dataclass
class CausalEdge:
    """Represents a causal relationship between two clauses"""
    source_id: str
    target_id: str
    relation_type: CausalRelationType
    confidence: float              # 0.0-1.0
    explanation: str               # Human-readable explanation
    
    def __repr__(self):
        return f"{self.source_id} --[{self.relation_type.value}]--> {self.target_id} (conf: {self.confidence:.2f})"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type.value,
            'confidence': self.confidence,
            'explanation': self.explanation
        }

class CLKGGraph:
    """
    Causal Legal Knowledge Graph - Production Implementation
    
    Nodes: Clauses with rich metadata
    Edges: Typed causal relationships with confidence scores
    """
    
    def __init__(self):
        self.clauses: Dict[str, Clause] = {}
        self.edges: List[CausalEdge] = []
        self.adjacency: Dict[str, List[str]] = {}  # clause_id -> list of connected clause_ids
        
        print("✅ CLKGGraph initialized")
    
    def add_clause(self, clause: Union[Clause, Dict]) -> None:
        """Add a clause node to the graph"""
        if isinstance(clause, dict):
            clause_obj = Clause(**clause)
        else:
            clause_obj = clause
        
        if clause_obj.id in self.clauses:
            print(f"⚠️ Clause {clause_obj.id} already exists, updating...")
            self.clauses[clause_obj.id] = clause_obj
        else:
            self.clauses[clause_obj.id] = clause_obj
            self.adjacency[clause_obj.id] = []
            
        print(f"  ✓ Added clause {clause_obj.id} (risk: {clause_obj.risk_score:.2f})")
    
    def add_edge(self, edge: Union[CausalEdge, Dict]) -> None:
        """Add a causal relationship edge"""
        if isinstance(edge, dict):
            edge_obj = CausalEdge(
                source_id=edge['source_id'],
                target_id=edge['target_id'],
                relation_type=CausalRelationType(edge['relation_type']),
                confidence=edge['confidence'],
                explanation=edge.get('explanation', '')
            )
        else:
            edge_obj = edge
        
        # Validate
        if edge_obj.source_id not in self.clauses:
            raise ValueError(f"Source clause {edge_obj.source_id} not in graph")
        if edge_obj.target_id not in self.clauses:
            raise ValueError(f"Target clause {edge_obj.target_id} not in graph")
        
        # Add edge
        self.edges.append(edge_obj)
        self.adjacency[edge_obj.source_id].append(edge_obj.target_id)
        
        print(f"  ✓ Added edge: {edge_obj}")
    
    def update_risk_scores(self, risk_scores: Dict[str, float]) -> None:
        """Update clause risk scores from GNN"""
        updated_count = 0
        for clause_id, risk_score in risk_scores.items():
            if clause_id in self.clauses:
                self.clauses[clause_id].risk_score = risk_score
                updated_count += 1
        
        print(f"  ✓ Updated {updated_count} clause risk scores")
    
    def get_neighbors(
        self,
        clause_id: str,
        relation_type: Optional[CausalRelationType] = None
    ) -> List[Clause]:
        """
        Get neighboring clauses
        
        Args:
            clause_id: ID of clause
            relation_type: Optional filter by relation type
            
        Returns:
            List of neighboring clauses
        """
        neighbors = []
        
        for edge in self.edges:
            if edge.source_id == clause_id:
                if relation_type is None or edge.relation_type == relation_type:
                    neighbors.append(self.clauses[edge.target_id])
        
        return neighbors
    
    def get_contradictions(self, clause_id: str) -> List[Clause]:
        """Get all clauses that contradict the given clause"""
        return self.get_neighbors(clause_id, CausalRelationType.CONTRADICTS)
    
    def get_supports(self, clause_id: str) -> List[Clause]:
        """Get all clauses that support the given clause"""
        return self.get_neighbors(clause_id, CausalRelationType.SUPPORTS)
    
    def find_contradiction_chains(self, max_length: int = 5) -> List[List[str]]:
        """
        Find chains of contradictions (cascade detection)
        
        Returns:
            List of clause ID chains that form contradiction paths
        """
        chains = []
        visited = set()
        
        def dfs(current_id: str, path: List[str]):
            if len(path) > max_length:
                return
            
            if current_id in visited:
                return
            
            visited.add(current_id)
            path.append(current_id)
            
            # Find contradictions
            contradictions = self.get_contradictions(current_id)
            for contradicted_clause in contradictions:
                if contradicted_clause.id not in path:
                    dfs(contradicted_clause.id, path.copy())
            
            # If path has multiple clauses, it's a chain
            if len(path) >= 2:
                chains.append(path[:])
        
        # Start DFS from each clause
        for clause_id in self.clauses:
            visited.clear()
            dfs(clause_id, [])
        
        print(f"🔍 Found {len(chains)} contradiction chains")
        return chains
    
    def get_statistics(self) -> Dict:
        """Get comprehensive graph statistics"""
        stats = {
            'num_clauses': len(self.clauses),
            'num_edges': len(self.edges),
            'avg_degree': len(self.edges) / len(self.clauses) if self.clauses else 0,
            'density': nx.density(nx.DiGraph([(e.source_id, e.target_id) for e in self.edges])) if self.edges else 0
        }
        
        # Relation type breakdown
        relation_counts = {}
        for edge in self.edges:
            rel_type = edge.relation_type.value
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        
        stats['relation_types'] = relation_counts
        stats['num_contradictions'] = relation_counts.get('CONTRADICTS', 0)
        stats['num_supports'] = relation_counts.get('SUPPORTS', 0)
        stats['avg_risk'] = np.mean([c.risk_score for c in self.clauses.values()]) if self.clauses else 0
        
        return stats
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for visualization"""
        G = nx.DiGraph()
        
        # Add nodes with rich attributes
        for clause_id, clause in self.clauses.items():
            G.add_node(
                clause_id,
                text=clause.text[:100] + "..." if len(clause.text) > 100 else clause.text,
                risk_score=clause.risk_score,
                confidence=clause.confidence,
                size=1.5 + clause.risk_score,  # Node size by risk
                color=clause.risk_score  # Node color by risk
            )
        
        # Add edges with attributes
        for edge in self.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                relation=edge.relation_type.value,
                confidence=edge.confidence,
                weight=edge.confidence,
                label=edge.relation_type.value[:3]  # Short label
            )
        
        return G
    
    def serialize(self, path: str = None) -> str:
        """Serialize graph to JSON"""
        data = {
            'clauses': {cid: clause.to_dict() for cid, clause in self.clauses.items()},
            'edges': [edge.to_dict() for edge in self.edges],
            'statistics': self.get_statistics()
        }
        
        if path:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Graph saved to {path}")
        
        return json.dumps(data, indent=2)
    
    @classmethod
    def deserialize(cls, data_or_path: Union[str, Dict]) -> 'CLKGGraph':
        """Load graph from JSON"""
        if isinstance(data_or_path, str):
            with open(data_or_path, 'r') as f:
                data = json.load(f)
        else:
            data = data_or_path
        
        graph = cls()
        
        # Load clauses
        for cid, clause_data in data['clauses'].items():
            graph.add_clause(clause_data)
        
        # Load edges
        for edge_data in data['edges']:
            graph.add_edge(edge_data)
        
        print(f"📂 Loaded graph: {len(graph.clauses)} clauses, {len(graph.edges)} edges")
        return graph
    
    def __repr__(self):
        stats = self.get_statistics()
        return f"CLKGGraph({stats['num_clauses']} clauses, {stats['num_edges']} edges)"

# Production usage example
if __name__ == "__main__":
    # Create sample graph
    graph = CLKGGraph()
    
    # Add clauses
    clause1 = Clause("C1", "This Agreement commences on execution date.", 0, 50)
    clause2 = Clause("C2", "Payment due within 30 days of invoice.", 100, 150)
    clause3 = Clause("C3", "Confidentiality obligation applies throughout term.", 200, 270)
    
    graph.add_clause(clause1)
    graph.add_clause(clause2)
    graph.add_clause(clause3)
    
    # Add causal edges
    edge1 = CausalEdge("C1", "C2", CausalRelationType.REQUIRES, 0.92, "Agreement start triggers payment obligations")
    edge2 = CausalEdge("C1", "C3", CausalRelationType.ENABLES, 0.88, "Agreement term enables confidentiality")
    
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    
    # Update risks
    graph.update_risk_scores({"C1": 0.3, "C2": 0.65, "C3": 0.82})
    
    # Get stats
    print("📊 Graph Statistics:")
    print(json.dumps(graph.get_statistics(), indent=2))
    
    # Find contradictions
    print("\n🔍 Contradiction chains:")
    chains = graph.find_contradiction_chains()
    for chain in chains:
        print("  →", " → ".join(chain))
    
    # NetworkX visualization ready
    nx_graph = graph.to_networkx()
    print(f"\n🎨 NetworkX graph ready: {nx_graph.number_of_nodes()} nodes")
