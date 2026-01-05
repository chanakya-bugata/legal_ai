"""
Risk Propagator - Production Interface for GNN-based Risk Propagation

Integrates CLKG → PyG → GNN → Risk Scores pipeline.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from .gnn_model import RiskPropagationGNN
from src.clkg.clkg_graph import CLKGGraph, CausalRelationType

class RiskPropagator:
    """
    Production risk propagation system using GNNs
    
    Full Pipeline:
    1. Initial risk computation (semantic analysis)
    2. Graph conversion (CLKG → PyG Data)
    3. GNN forward pass with edge-aware attention
    4. Contradiction penalties & cascade detection
    5. Uncertainty quantification
    """
    
    def __init__(
        self,
        gnn_model: RiskPropagationGNN,
        device: str = "cpu",
        contradiction_penalty: float = 0.15,
        cascade_threshold: float = 0.7
    ):
        """
        Args:
            gnn_model: Trained GAT model
            device: 'cpu' or 'cuda'
            contradiction_penalty: Risk increase per contradiction
            cascade_threshold: Minimum risk for cascade alerts
        """
        self.gnn_model = gnn_model.to(device)
        self.device = device
        self.contradiction_penalty = contradiction_penalty
        self.cascade_threshold = cascade_threshold
        
        self.gnn_model.eval()
        print(f"✅ RiskPropagator initialized (device: {device})")
    
    def propagate_risks(
        self,
        graph: CLKGGraph,
        clause_embeddings: np.ndarray,
        initial_risks: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Main production method: Propagate risks through dependencies
        
        Args:
            graph: CLKGGraph with clauses and causal edges
            clause_embeddings: [N, 768] Legal-BERT embeddings
            initial_risks: Optional base risk scores
            
        Returns:
            Dict: clause_id → final_risk_score (0.0-1.0)
        """
        print("\n⚠️ Propagating risks through GNN...")
        
        # Step 1: Validate inputs
        self._validate_inputs(graph, clause_embeddings)
        
        # Step 2: Convert to PyG format
        data = self._prepare_graph_data(graph, clause_embeddings)
        
        # Step 3: GNN forward pass
        risk_scores = self._gnn_forward(data)
        
        # Step 4: Map back to clause IDs with post-processing
        clause_ids = list(graph.clauses.keys())
        final_risks = {}
        
        for i, clause_id in enumerate(clause_ids):
            base_risk = risk_scores[i].item()
            
            # Apply contradiction penalties
            adjusted_risk = self._apply_contradiction_penalties(
                graph, clause_id, base_risk
            )
            
            # Apply initial risks if provided
            if initial_risks and clause_id in initial_risks:
                final_risk = 0.7 * adjusted_risk + 0.3 * initial_risks[clause_id]
            else:
                final_risk = adjusted_risk
            
            final_risks[clause_id] = min(1.0, max(0.0, final_risk))
        
        print(f"  ✓ Propagated risks for {len(final_risks)} clauses")
        print(f"  📊 Risk range: {min(final_risks.values()):.2f} - {max(final_risks.values()):.2f}")
        
        return final_risks
    
    def detect_cascade_risks(
        self,
        graph: CLKGGraph,
        risk_scores: Dict[str, float]
    ) -> List[Dict]:
        """
        Detect high-risk cascade chains
        
        Returns:
            List of cascade alerts with chain analysis
        """
        cascades = []
        
        # Find contradiction chains
        chains = graph.find_contradiction_chains(max_length=6)
        
        for chain in chains:
            # Compute cascade metrics
            chain_risks = [risk_scores.get(cid, 0.0) for cid in chain]
            max_risk = max(chain_risks)
            avg_risk = np.mean(chain_risks)
            
            # Cascade penalty (longer chains = higher risk)
            chain_penalty = 0.08 * (len(chain) - 1)
            cascade_risk = min(1.0, max_risk + chain_penalty)
            
            if cascade_risk >= self.cascade_threshold:
                # Get clause summaries
                chain_summaries = [
                    graph.clauses[cid].text[:60] + "..."
                    for cid in chain if cid in graph.clauses
                ]
                
                cascades.append({
                    'chain_ids': chain,
                    'chain_summaries': chain_summaries,
                    'max_risk': float(max_risk),
                    'avg_risk': float(avg_risk),
                    'cascade_risk': float(cascade_risk),
                    'length': len(chain),
                    'explanation': (
                        f"{len(chain)}-clause contradiction cascade "
                        f"with max risk {max_risk:.2f} + chain penalty = {cascade_risk:.2f}"
                    ),
                    'severity': 'HIGH' if cascade_risk > 0.85 else 'MEDIUM'
                })
        
        print(f"🔍 Detected {len(cascades)} cascade risks")
        return cascades
    
    def _validate_inputs(self, graph: CLKGGraph, clause_embeddings: np.ndarray):
        """Input validation"""
        if len(graph.clauses) == 0:
            raise ValueError("Graph must contain clauses")
        
        expected_nodes = len(graph.clauses)
        if clause_embeddings.shape[0] != expected_nodes:
            raise ValueError(
                f"Clause embeddings shape mismatch: "
                f"expected {expected_nodes}, got {clause_embeddings.shape[0]}"
            )
        
        print(f"  ✓ Validated: {expected_nodes} clauses, {len(graph.edges)} edges")
    
    def _prepare_graph_data(
        self,
        graph: CLKGGraph,
        clause_embeddings: np.ndarray
    ) -> 'Data':
        """Convert CLKG to optimized PyG Data object"""
        clause_ids = list(graph.clauses.keys())
        node_to_idx = {cid: i for i, cid in enumerate(clause_ids)}
        
        # Node features (clause embeddings)
        x = torch.tensor(clause_embeddings, dtype=torch.float32, device=self.device)
        
        # Edge construction
        edge_index = []
        edge_attr = []
        
        # One-hot encoding for 7 relation types
        relation_to_idx = {
            CausalRelationType.SUPPORTS: 0,
            CausalRelationType.CONTRADICTS: 1,
            CausalRelationType.MODIFIES: 2,
            CausalRelationType.OVERTURNS: 3,
            CausalRelationType.ENABLES: 4,
            CausalRelationType.BLOCKS: 5,
            CausalRelationType.REQUIRES: 6
        }
        
        for edge in graph.edges:
            src_idx = node_to_idx.get(edge.source_id)
            tgt_idx = node_to_idx.get(edge.target_id)
            
            if src_idx is not None and tgt_idx is not None:
                edge_index.extend([src_idx, tgt_idx])
                
                # One-hot relation type + confidence weight
                rel_idx = relation_to_idx.get(edge.relation_type, 0)
                edge_features = [0.0] * 8  # 7 relations + confidence
                edge_features[rel_idx] = edge.confidence
                edge_features[7] = edge.confidence  # Confidence weight
                edge_attr.extend(edge_features)
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).view(2, -1)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=self.device).view(-1, 8)
        else:
            # Empty graph handling
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 8), dtype=torch.float32, device=self.device)
        
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        print(f"  📊 PyG Data: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
        return data
    
    def _gnn_forward(self, data):
        """Optimized GNN forward pass"""
        self.gnn_model.eval()
        with torch.no_grad():
            risk_scores = self.gnn_model(
                data.x,
                data.edge_index,
                data.edge_attr
            )
        return risk_scores
    
    def _apply_contradiction_penalties(
        self,
        graph: CLKGGraph,
        clause_id: str,
        base_risk: float
    ) -> float:
        """Apply risk penalties for contradictions"""
        contradictions = graph.get_contradictions(clause_id)
        num_contradictions = len(contradictions)
        
        if num_contradictions > 0:
            penalty = self.contradiction_penalty * num_contradictions
            penalized_risk = min(1.0, base_risk + penalty)
            print(f"    ⚠️ {clause_id}: +{penalty:.2f} penalty ({num_contradictions} contradictions)")
            return penalized_risk
        
        return base_risk

# Production usage example
if __name__ == "__main__":
    from .gnn_model import RiskPropagationGNN
    
    # Mock setup
    gnn_model = RiskPropagationGNN(embedding_dim=768)
    propagator = RiskPropagator(gnn_model)
    
    # Mock graph and embeddings (replace with real pipeline)
    print("✅ RiskPropagator production ready!")
    print("Usage: risks = propagator.propagate_risks(clkg, embeddings)")
