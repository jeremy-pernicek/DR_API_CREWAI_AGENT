"""
Hybrid retrieval system combining BM25 keyword search with FAISS semantic search.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import pickle

from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    """A document in our search corpus"""
    id: str
    title: str
    content: str
    type: str  # 'method', 'example', 'tutorial', etc.
    source: str  # 'python_sdk', 'rest_api', 'github'
    url: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class SearchResult:
    """A search result with score and source"""
    document: Document
    score: float
    source: str  # 'bm25', 'semantic', 'hybrid'


class HybridRetriever:
    """Hybrid search system combining keyword and semantic search"""
    
    def __init__(self, 
                 data_dir: str = "agent_crewai/data",
                 model_name: str = "all-MiniLM-L6-v2",
                 bm25_weight: float = 0.6,
                 semantic_weight: float = 0.4):
        
        self.data_dir = Path(data_dir)
        self.indexes_dir = self.data_dir / "indexes"
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        
        # Initialize components
        self.documents: List[Document] = []
        self.bm25_index = None
        self.semantic_index = None
        self.embeddings = None
        
        # Initialize sentence transformer
        print(f"Loading sentence transformer model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Try to load existing indexes
        self.load_indexes()
    
    def add_documents(self, docs: List[Document]) -> None:
        """Add documents to the retriever"""
        self.documents.extend(docs)
        print(f"Added {len(docs)} documents. Total: {len(self.documents)}")
    
    def build_indexes(self) -> None:
        """Build both BM25 and semantic indexes"""
        if not self.documents:
            print("No documents to index")
            return
        
        print(f"Building indexes for {len(self.documents)} documents...")
        
        # Build BM25 index
        print("Building BM25 index...")
        tokenized_docs = [doc.content.lower().split() for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
        
        # Build semantic index
        print("Building semantic index...")
        texts = [f"{doc.title} {doc.content}" for doc in self.documents]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.semantic_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.semantic_index.add(self.embeddings)
        
        print("Indexes built successfully")
        self.save_indexes()
    
    def search(self, query: str, top_k: int = 10, source: str = "hybrid") -> List[SearchResult]:
        """
        Search documents using specified method.
        
        Args:
            query: Search query
            top_k: Number of results to return
            source: 'bm25', 'semantic', or 'hybrid'
        """
        if not self.documents:
            return []
        
        if source == "bm25":
            return self._bm25_search(query, top_k)
        elif source == "semantic":
            return self._semantic_search(query, top_k)
        elif source == "hybrid":
            return self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"Unknown search source: {source}")
    
    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """BM25 keyword search"""
        if self.bm25_index is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return relevant results
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                    source="bm25"
                ))
        
        return results
    
    def _semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Semantic vector search"""
        if self.semantic_index is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.semantic_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(score),
                    source="semantic"
                ))
        
        return results
    
    def _hybrid_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Hybrid search combining BM25 and semantic results"""
        # Get results from both methods
        bm25_results = self._bm25_search(query, top_k * 2)  # Get more to ensure diversity
        semantic_results = self._semantic_search(query, top_k * 2)
        
        # Normalize scores to 0-1 range
        if bm25_results:
            max_bm25 = max(r.score for r in bm25_results)
            for result in bm25_results:
                result.score = result.score / max_bm25 if max_bm25 > 0 else 0
        
        if semantic_results:
            max_semantic = max(r.score for r in semantic_results)
            for result in semantic_results:
                result.score = result.score / max_semantic if max_semantic > 0 else 0
        
        # Combine results with weighted scores
        combined_scores = {}
        
        for result in bm25_results:
            doc_id = result.document.id
            combined_scores[doc_id] = {
                'document': result.document,
                'bm25_score': result.score * self.bm25_weight,
                'semantic_score': 0,
                'total_score': result.score * self.bm25_weight
            }
        
        for result in semantic_results:
            doc_id = result.document.id
            if doc_id in combined_scores:
                combined_scores[doc_id]['semantic_score'] = result.score * self.semantic_weight
                combined_scores[doc_id]['total_score'] += result.score * self.semantic_weight
            else:
                combined_scores[doc_id] = {
                    'document': result.document,
                    'bm25_score': 0,
                    'semantic_score': result.score * self.semantic_weight,
                    'total_score': result.score * self.semantic_weight
                }
        
        # Sort by combined score and return top k
        sorted_results = sorted(combined_scores.values(), 
                              key=lambda x: x['total_score'], 
                              reverse=True)
        
        results = []
        for item in sorted_results[:top_k]:
            results.append(SearchResult(
                document=item['document'],
                score=item['total_score'],
                source="hybrid"
            ))
        
        return results
    
    def save_indexes(self) -> None:
        """Save indexes to disk"""
        try:
            # Save BM25 index
            if self.bm25_index:
                with open(self.indexes_dir / "bm25_index.pkl", "wb") as f:
                    pickle.dump(self.bm25_index, f)
            
            # Save semantic index
            if self.semantic_index:
                faiss.write_index(self.semantic_index, str(self.indexes_dir / "semantic_index.faiss"))
            
            # Save embeddings
            if self.embeddings is not None:
                np.save(self.indexes_dir / "embeddings.npy", self.embeddings)
            
            # Save documents
            docs_data = []
            for doc in self.documents:
                docs_data.append({
                    'id': doc.id,
                    'title': doc.title,
                    'content': doc.content,
                    'type': doc.type,
                    'source': doc.source,
                    'url': doc.url,
                    'metadata': doc.metadata
                })
            
            with open(self.indexes_dir / "documents.json", "w", encoding='utf-8') as f:
                json.dump(docs_data, f, indent=2, ensure_ascii=False)
            
            print(f"Indexes saved to {self.indexes_dir}")
            
        except Exception as e:
            print(f"Error saving indexes: {e}")
    
    def load_indexes(self) -> None:
        """Load indexes from disk"""
        try:
            # Load documents
            docs_file = self.indexes_dir / "documents.json"
            if docs_file.exists():
                with open(docs_file, "r", encoding='utf-8') as f:
                    docs_data = json.load(f)
                
                self.documents = []
                for doc_data in docs_data:
                    self.documents.append(Document(**doc_data))
                
                print(f"Loaded {len(self.documents)} documents")
            
            # Load BM25 index
            bm25_file = self.indexes_dir / "bm25_index.pkl"
            if bm25_file.exists():
                with open(bm25_file, "rb") as f:
                    self.bm25_index = pickle.load(f)
                print("Loaded BM25 index")
            
            # Load semantic index
            semantic_file = self.indexes_dir / "semantic_index.faiss"
            if semantic_file.exists():
                self.semantic_index = faiss.read_index(str(semantic_file))
                print("Loaded semantic index")
            
            # Load embeddings
            embeddings_file = self.indexes_dir / "embeddings.npy"
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
                print(f"Loaded embeddings: {self.embeddings.shape}")
            
        except Exception as e:
            print(f"Error loading indexes: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            'total_documents': len(self.documents),
            'has_bm25_index': self.bm25_index is not None,
            'has_semantic_index': self.semantic_index is not None,
            'bm25_weight': self.bm25_weight,
            'semantic_weight': self.semantic_weight,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0
        }


def create_sample_data() -> List[Document]:
    """Create some sample DataRobot documentation for testing"""
    return [
        Document(
            id="project_create",
            title="Create DataRobot Project",
            content="To create a DataRobot project, use dr.Project.start(dataset_path, target='target_column'). This method uploads your dataset and creates a new project for machine learning modeling.",
            type="method",
            source="python_sdk",
            url="https://docs.datarobot.com/project-create"
        ),
        Document(
            id="autopilot_start", 
            title="Start Autopilot",
            content="After creating a project, start Autopilot with project.set_target() and project.wait_for_autopilot(). Autopilot automatically builds and evaluates multiple machine learning models.",
            type="method",
            source="python_sdk", 
            url="https://docs.datarobot.com/autopilot"
        ),
        Document(
            id="timeseries_example",
            title="Time Series Project Example",
            content="For time series modeling, set datetime_partition_column and feature derivation windows. Example: project.set_target(target='sales', partition_settings={'datetime_partition_column': 'date', 'forecast_window_start': 1, 'forecast_window_end': 7})",
            type="example",
            source="python_sdk",
            url="https://docs.datarobot.com/timeseries-example"
        ),
        Document(
            id="deployment_create",
            title="Create Deployment", 
            content="Deploy a model using dr.Deployment.create_from_learning_model(model_id, label='My Deployment'). This creates a prediction endpoint for your trained model.",
            type="method",
            source="python_sdk",
            url="https://docs.datarobot.com/deployment-create"
        ),
        Document(
            id="batch_predictions",
            title="Make Batch Predictions",
            content="Use deployment.predict_batch(dataset_path) to make predictions on a batch of data. For time series, include timeseries_settings with forecast_point parameter.",
            type="method", 
            source="python_sdk",
            url="https://docs.datarobot.com/batch-predictions"
        )
    ]


if __name__ == "__main__":
    # Test the hybrid retriever
    retriever = HybridRetriever()
    
    # Add sample data
    sample_docs = create_sample_data()
    retriever.add_documents(sample_docs)
    
    # Build indexes
    retriever.build_indexes()
    
    # Test searches
    test_queries = [
        "create a DataRobot project",
        "time series forecasting",
        "make predictions on new data"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = retriever.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.document.title} (score: {result.score:.3f})")
            print(f"     {result.document.content[:100]}...")
    
    print(f"\n📊 Stats: {retriever.get_stats()}")