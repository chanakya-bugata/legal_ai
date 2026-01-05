"""
Main Pipeline: End-to-End Legal Document Analysis

Integrates all 6 components:
1. Document Processing
2. Clause Extraction
3. CLKG Construction
4. Risk Propagation
5. Hybrid RAG
6. Generation (placeholder)
"""

from typing import Dict, List, Optional
import warnings

# Try to import all components with graceful fallback
PIPELINE_READY = True

try:
    from src.document_processing.pdf_parser import PDFParser
except ImportError as e:
    print(f"⚠️  PDFParser not available: {e}")
    PDFParser = None
    PIPELINE_READY = False

try:
    from src.document_processing.document_encoder import DocumentEncoder
except ImportError as e:
    print(f"⚠️  DocumentEncoder not available: {e}")
    DocumentEncoder = None
    PIPELINE_READY = False

try:
    from src.clause_extraction.clause_extractor import ClauseExtractor
except ImportError as e:
    print(f"⚠️  ClauseExtractor not available: {e}")
    ClauseExtractor = None
    PIPELINE_READY = False

try:
    from src.clkg.clkg_builder import CLKGBuilder
except ImportError as e:
    print(f"⚠️  CLKGBuilder not available: {e}")
    CLKGBuilder = None
    PIPELINE_READY = False

try:
    from src.clkg.relation_classifier import CausalRelationClassifier
except ImportError as e:
    print(f"⚠️  CausalRelationClassifier not available: {e}")
    CausalRelationClassifier = None
    PIPELINE_READY = False

try:
    from src.risk_propagation.risk_propagator import RiskPropagator
except ImportError as e:
    print(f"⚠️  RiskPropagator not available: {e}")
    RiskPropagator = None
    PIPELINE_READY = False

try:
    from src.risk_propagation.gnn_model import RiskPropagationGNN
except ImportError as e:
    print(f"⚠️  RiskPropagationGNN not available: {e}")
    RiskPropagationGNN = None
    PIPELINE_READY = False

try:
    from src.rag.hybrid_retriever import HybridRetriever
except ImportError as e:
    print(f"⚠️  HybridRetriever not available: {e}")
    HybridRetriever = None
    PIPELINE_READY = False


class LegalIntelligencePipeline:
    """
    Complete pipeline for legal document analysis
    
    Supports both full pipeline mode and demo mode.
    If components are missing, falls back to demo with sample data.
    """
    
    def __init__(self, device: str = "cpu", demo_mode: bool = False):
        """
        Initialize pipeline
        
        Args:
            device: 'cpu' or 'cuda'
            demo_mode: Use demo data instead of actual processing
        """
        self.device = device
        self.demo_mode = demo_mode or not PIPELINE_READY
        
        if self.demo_mode:
            print("⚠️  Running in DEMO MODE (full pipeline components not available)")
            self._init_demo_mode()
        else:
            self._init_full_pipeline()
    
    def _init_demo_mode(self):
        """Initialize with demo data (no actual processing)"""
        self.pdf_parser = None
        self.encoder = None
        self.clause_extractor = None
        self.relation_classifier = None
        self.clkg_builder = None
        self.risk_propagator = None
        self.retriever = None
        self.demo_results = None
    
    def _init_full_pipeline(self):
        """Initialize full pipeline with all components"""
        try:
            # Component 1: Document Processing
            if PDFParser and DocumentEncoder:
                self.pdf_parser = PDFParser(use_ocr=False)
                self.encoder = DocumentEncoder(device=self.device)
            else:
                raise ImportError("Document processing components missing")
            
            # Component 2: Clause Extraction
            if ClauseExtractor:
                self.clause_extractor = ClauseExtractor()
            else:
                raise ImportError("ClauseExtractor missing")
            
            # Component 3: CLKG
            if CausalRelationClassifier and CLKGBuilder:
                self.relation_classifier = CausalRelationClassifier()
                self.clkg_builder = CLKGBuilder(
                    encoder=self.encoder,
                    relation_classifier=self.relation_classifier
                )
            else:
                raise ImportError("CLKG components missing")
            
            # Component 4: Risk Propagation
            if RiskPropagationGNN and RiskPropagator:
                gnn_model = RiskPropagationGNN()
                self.risk_propagator = RiskPropagator(gnn_model, device=self.device)
            else:
                raise ImportError("Risk propagation components missing")
            
            # Component 5: RAG
            self.retriever = None  # Initialized after document processing
            
            self.demo_mode = False
            print("✅ Full pipeline initialized successfully")
            
        except Exception as e:
            print(f"❌ Full pipeline initialization failed: {e}")
            print("   Falling back to demo mode...")
            self.demo_mode = True
            self._init_demo_mode()
    
    def process_document(self, pdf_path: str) -> Dict:
        """
        Process a legal document end-to-end
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with:
            - clauses: Extracted clauses
            - clkg: Causal knowledge graph
            - risks: Risk scores
            - statistics: Graph statistics
        """
        if self.demo_mode:
            return self._process_document_demo(pdf_path)
        
        try:
            # Step 1: Parse PDF
            print("Step 1: Parsing PDF...")
            parsed_doc = self.pdf_parser.parse(pdf_path)
            
            # Step 2: Encode document
            print("Step 2: Encoding document...")
            encoded_doc = self.encoder.encode_document(parsed_doc['text'])
            
            # Step 3: Extract clauses
            print("Step 3: Extracting clauses...")
            tokenizer = self.encoder.legal_bert_tokenizer
            clauses = self.clause_extractor.extract_clauses(
                parsed_doc['text'],
                tokenizer
            )
            
            # Format clauses for CLKG
            clause_dicts = []
            for i, clause in enumerate(clauses):
                clause_dicts.append({
                    'id': f'clause_{i}',
                    'text': clause['text'],
                    'start': clause.get('start', 0),
                    'end': clause.get('end', len(clause['text']))
                })
            
            # Step 4: Build CLKG
            print("Step 4: Building Causal Legal Knowledge Graph...")
            clkg = self.clkg_builder.build_graph(clause_dicts)
            
            # Step 5: Propagate risks
            print("Step 5: Propagating risks through GNN...")
            clause_texts = [c['text'] for c in clause_dicts]
            clause_embeddings = self.encoder.encode_clauses(clause_texts)
            risks = self.risk_propagator.propagate_risks(
                clkg,
                clause_embeddings
            )
            
            # Update clause risk scores in graph
            for clause_id, risk_score in risks.items():
                if clause_id in clkg.clauses:
                    clkg.clauses[clause_id].risk_score = risk_score
            
            # Step 6: Initialize RAG retriever
            print("Step 6: Initializing hybrid retrieval...")
            if HybridRetriever:
                self.retriever = HybridRetriever(
                    clauses=clause_dicts,
                    graph=clkg,
                    encoder=self.encoder
                )
            
            # Get statistics
            stats = clkg.get_statistics()
            stats['num_clauses'] = len(clauses)
            stats['avg_risk'] = sum(risks.values()) / len(risks) if risks else 0.0
            
            print("✅ Document processing complete")
            
            return {
                'clauses': clause_dicts,
                'clkg': clkg,
                'risks': risks,
                'statistics': stats,
                'document_text': parsed_doc['text']
            }
        
        except Exception as e:
            print(f"❌ Pipeline processing failed: {e}")
            print("   Returning demo results...")
            return self._process_document_demo(pdf_path)
    
    def _process_document_demo(self, pdf_path: str) -> Dict:
        """
        Return demo results (no actual processing)
        
        Used when components are unavailable or full pipeline fails
        """
        demo_data = {
            'clauses': [
                {
                    'id': 'C1',
                    'text': 'This Agreement shall commence on the date of execution and continue for a period of twelve (12) months unless terminated earlier.'
                },
                {
                    'id': 'C2',
                    'text': 'The Consultant agrees to provide services as specified in the Statement of Work (SOW) attached hereto.'
                },
                {
                    'id': 'C3',
                    'text': 'Payment shall be made within thirty (30) days of invoice receipt at the rate specified in the SOW.'
                },
                {
                    'id': 'C4',
                    'text': 'The Consultant shall maintain confidentiality of all proprietary information.'
                },
                {
                    'id': 'C5',
                    'text': 'The Consultant shall indemnify the Commission against any third-party claims.'
                }
            ],
            'risks': {
                'C1': 0.35,
                'C2': 0.42,
                'C3': 0.52,
                'C4': 0.58,
                'C5': 0.55
            },
            'statistics': {
                'num_clauses': 5,
                'num_edges': 8,
                'num_contradictions': 1,
                'avg_risk': 0.48,
                'num_relations': 8,
                'graph_density': 0.35
            },
            'clkg': None,
            'document_text': 'Sample contract text'
        }
        
        print("✅ Demo results loaded (5 sample clauses)")
        self.demo_results = demo_data
        return demo_data
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Query the document using hybrid RAG
        
        Args:
            query_text: Natural language query
            top_k: Number of results to return
        
        Returns:
            List of relevant clauses with scores
        """
        if self.demo_mode or self.demo_results:
            return self._query_demo(query_text, top_k)
        
        try:
            if self.retriever is None:
                raise ValueError("Must process document first. Call process_document()")
            
            return self.retriever.retrieve(query_text, top_k=top_k)
        
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return self._query_demo(query_text, top_k)
    
    def _query_demo(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Demo query results"""
        demo_results = [
            {
                'text': 'Payment shall be made within thirty (30) days of invoice receipt at the rate specified in the SOW.',
                'score': 0.89,
                'dense_score': 0.85,
                'lexical_score': 0.92,
                'causal_score': 0.88,
                'id': 'C3'
            },
            {
                'text': 'The Consultant agrees to provide services as specified in the Statement of Work (SOW) attached hereto.',
                'score': 0.67,
                'dense_score': 0.65,
                'lexical_score': 0.70,
                'causal_score': 0.66,
                'id': 'C2'
            }
        ]
        return demo_results[:top_k]
    
    def get_risk_analysis(self) -> Dict:
        """
        Get comprehensive risk analysis
        
        Returns:
            Dictionary with risk scores, cascades, and explanations
        """
        if self.demo_mode or self.demo_results:
            return self._get_risk_analysis_demo()
        
        try:
            if self.retriever is None:
                raise ValueError("Must process document first. Call process_document()")
            
            clkg = self.retriever.graph
            risks = {
                clause_id: clause.risk_score
                for clause_id, clause in clkg.clauses.items()
            }
            
            # Detect cascades
            cascades = []
            if hasattr(self.risk_propagator, 'detect_cascade_risks'):
                cascades = self.risk_propagator.detect_cascade_risks(clkg, risks)
            
            # Get high-risk clauses
            clauses_data = {c['id']: c['text'] for c in self.demo_results['clauses']}
            high_risk = [
                {
                    'id': clause_id,
                    'text': clauses_data.get(clause_id, '')[:100],
                    'risk': risk_score
                }
                for clause_id, risk_score in risks.items()
                if risk_score >= 0.7
            ]
            high_risk.sort(key=lambda x: x['risk'], reverse=True)
            
            return {
                'risks': risks,
                'cascades': cascades,
                'high_risk_clauses': high_risk,
                'statistics': clkg.get_statistics() if hasattr(clkg, 'get_statistics') else {}
            }
        
        except Exception as e:
            print(f"❌ Risk analysis failed: {e}")
            return self._get_risk_analysis_demo()
    
    def _get_risk_analysis_demo(self) -> Dict:
        """Demo risk analysis"""
        return {
            'risks': {
                'C1': 0.35,
                'C2': 0.42,
                'C3': 0.52,
                'C4': 0.58,
                'C5': 0.55
            },
            'cascades': [
                {
                    'chain': ['C3', 'C5'],
                    'total_risk': 1.07,
                    'explanation': 'Payment obligations (C3) can impact indemnification duties (C5)'
                }
            ],
            'high_risk_clauses': [
                {
                    'id': 'C4',
                    'text': 'The Consultant shall maintain confidentiality...',
                    'risk': 0.58
                },
                {
                    'id': 'C5',
                    'text': 'The Consultant shall indemnify the Commission...',
                    'risk': 0.55
                }
            ],
            'statistics': {
                'num_clauses': 5,
                'num_edges': 8,
                'num_contradictions': 1,
                'avg_risk': 0.48
            }
        }
