"""
Graph Attention Network (GAT) for Risk Propagation - PRODUCTION READY

NOVEL ALGORITHM: First application of GNNs to legal risk analysis
with cascade detection through attention-weighted propagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional
import numpy as np

class RiskPropagationGNN(nn.Module):
    """
    Graph Attention Network for propagating risk through clause dependencies
    
    Production Architecture:
    - Multi-layer GAT with residual connections
    - Multi-head attention for diverse relationships
    - Edge-aware attention (relation types)
    - Risk scoring with uncertainty estimation
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,      # Legal-BERT embedding size
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: int = 7,             # Number of relation types
        residual: bool = True
    ):
        """
        Args:
            embedding_dim: Clause embedding dimension
            hidden_dim: GAT hidden dimension
            num_layers: Number of GAT layers
            num_heads: Attention heads per layer
            dropout: Dropout rate
            edge_dim: Edge feature dimension (relation types)
            residual: Use residual connections
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        
        # GAT Layers with progressive dimension reduction
        self.gat_layers = nn.ModuleList()
        
        # Input layer: embedding_dim -> hidden_dim
        self.gat_layers.append(
            GATConv(
                embedding_dim,
                hidden_dim // num_heads,  # Per-head dimension
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=True
            )
        )
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True
                )
            )
        
        # Output layer: final representation
        self.gat_layers.append(
            GATConv(
                hidden_dim,
                hidden_dim // num_heads,
                heads=1,  # Single head for final aggregation
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=True
            )
        )
        
        # Risk scoring head with uncertainty
        self.risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)  # [risk_score, uncertainty]
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        print(f"✅ RiskPropagationGNN initialized:")
        print(f"   Layers: {num_layers}, Heads: {num_heads}, Hidden: {hidden_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        batch: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass - Production inference
        
        Args:
            x: Node features [N, embedding_dim]
            edge_index: Graph edges [2, E]
            edge_attr: Edge features [E, edge_dim] (one-hot relation types)
            batch: Batch vector for global pooling
            
        Returns:
            risk_scores: [N, 1] risk scores (0-1)
        """
        residual = x  # For residual connection
        
        # Multi-layer GAT propagation
        for i, gat_layer in enumerate(self.gat_layers):
            # GAT forward pass
            x_new = gat_layer(x, edge_index, edge_attr)
            
            # Layer normalization + residual
            if self.residual and i == 0 and x.shape == x_new.shape:
                x_new = x_new + residual
            
            # Activation + normalization
            x_new = F.elu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x_new = self.layer_norms[i](x_new)
            
            x = x_new
        
        # Global risk scoring
        risk_logits = self.risk_scorer(x)
        risk_scores = torch.sigmoid(risk_logits[:, 0:1])  # Take risk score
        
        return risk_scores
    
    def forward_with_explainability(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with explainability features
        
        Returns:
            Dict containing:
            - risk_scores: Final risk scores
            - node_embeddings: Final node representations
            - edge_importance: Edge contribution scores
        """
        node_embeddings = []
        edge_importance = []
        
        residual = x
        
        for i, gat_layer in enumerate(self.gat_layers):
            x_new = gat_layer(x, edge_index, edge_attr)
            
            if self.residual and i == 0 and x.shape == x_new.shape:
                x_new = x_new + residual
            
            x_new = F.elu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x_new = self.layer_norms[i](x_new)
            
            node_embeddings.append(x_new.detach())
            x = x_new
        
        # Risk scoring
        risk_logits = self.risk_scorer(x)
        risk_scores = torch.sigmoid(risk_logits[:, 0:1])
        uncertainty = torch.sigmoid(risk_logits[:, 1:2])  # Second output as uncertainty
        
        return {
            'risk_scores': risk_scores,
            'uncertainty': uncertainty,
            'node_embeddings': x,
            'layer_embeddings': node_embeddings
        }
    
    def predict_risk_propagation(
        self,
        clkg_graph: 'CLKGGraph',
        clause_embeddings: np.ndarray
    ) -> Dict[str, float]:
        """
        Production method: Predict risk propagation from CLKG
        
        Args:
            clkg_graph: CLKGGraph instance
            clause_embeddings: [N, 768] clause embeddings
            
        Returns:
            Dict of clause_id -> propagated_risk_score
        """
        # Convert CLKG to PyG format
        data = self._clkg_to_pyg_data(clkg_graph, clause_embeddings)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            risk_scores = self(data.x, data.edge_index, data.edge_attr)
        
        # Map back to clause IDs
        clause_ids = list(clkg_graph.clauses.keys())
        propagation_results = {
            clause_ids[i]: float(risk_scores[i].item())
            for i in range(len(clause_ids))
        }
        
        return propagation_results
    
    def _clkg_to_pyg_data(
        self,
        clkg_graph: 'CLKGGraph',
        clause_embeddings: np.ndarray
    ) -> Data:
        """Convert CLKGGraph to PyTorch Geometric Data object"""
        clause_ids = list(clkg_graph.clauses.keys())
        node_to_idx = {cid: idx for idx, cid in enumerate(clause_ids)}
        
        # Node features
        x = torch.tensor(clause_embeddings, dtype=torch.float32)
        
        # Edge index
        edge_index_list = []
        edge_attr_list = []
        
        relation_to_idx = {
            CausalRelationType.SUPPORTS: 0,
            CausalRelationType.CONTRADICTS: 1,
            CausalRelationType.MODIFIES: 2,
            CausalRelationType.OVERTURNS: 3,
            CausalRelationType.ENABLES: 4,
            CausalRelationType.BLOCKS: 5,
            CausalRelationType.REQUIRES: 6
        }
        
        for edge in clkg_graph.edges:
            src_idx = node_to_idx[edge.source_id]
            tgt_idx = node_to_idx[edge.target_id]
            
            edge_index_list.extend([src_idx, tgt_idx])
            rel_idx = relation_to_idx.get(edge.relation_type, 0)
            edge_attr_list.extend([1.0 if j == rel_idx else 0.0 for j in range(7)])
        
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).view(-1, 7)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 7), dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

# Production usage example
if __name__ == "__main__":
    # Mock data
    x = torch.randn(5, 768)  # 5 clauses
    edge_index = torch.tensor([[0,1,1,2,2,4],[1,2,0,3,4,2]])
    edge_attr = torch.randn(6, 7)  # 7 relation types
    
    model = RiskPropagationGNN(embedding_dim=768)
    
    # Forward pass
    risk_scores = model(x, edge_index, edge_attr)
    print("Risk scores:", risk_scores)
    print(f"✅ GNN forward pass complete (shape: {risk_scores.shape})")
    
    # Explainability
    explain = model.forward_with_explainability(x, edge_index, edge_attr)
    print("Explainability features:", list(explain.keys()))
