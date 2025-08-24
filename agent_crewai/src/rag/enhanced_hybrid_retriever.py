"""
Enhanced Hybrid Retriever with Multi-Index Architecture
Supports multiple content types with specialized retrieval strategies for code generation.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import pickle
from collections import defaultdict
import logging

from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSearchResult:
    """Enhanced search result with rich metadata"""
    id: str
    title: str
    content: str
    content_type: str  # 'code', 'documentation', 'workflow', 'template'
    source_type: str   # 'readthedocs', 'github_python', 'github_notebook', etc.
    source_file: str
    tags: List[str]
    code_examples: List[str]
    api_methods: List[str]
    complexity_score: float
    use_case_category: str
    score: float
    search_source: str  # 'bm25', 'semantic', 'hybrid'
    metadata: Dict[str, Any]

class EnhancedHybridRetriever:
    """Enhanced hybrid retriever with multi-index architecture"""
    
    def __init__(self, 
                 data_dir: str = "agent_crewai/data",
                 model_name: str = "all-MiniLM-L6-v2",
                 index_weights: Dict[str, float] = None):
        
        self.data_dir = Path(data_dir)
        self.scraped_dir = self.data_dir / "scraped"
        self.indexes_dir = self.data_dir / "indexes"
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        
        # Default index weights for different content types
        self.index_weights = index_weights or {
            'code': 1.2,           # Prioritize working code examples
            'workflow': 1.1,       # Notebook workflows are valuable
            'template': 1.0,       # Templates are good references
            'documentation': 0.8   # Documentation is helpful but less actionable
        }
        
        # Source type weights
        self.source_weights = {
            'github_python': 1.3,     # Real working code gets highest priority
            'github_notebook': 1.2,   # Comprehensive examples
            'ai_accelerator': 1.1,    # Templates and integrations
            'readthedocs': 1.0,       # Official documentation
            'openapi': 0.9            # API specs are reference material
        }
        
        # Multi-index architecture
        self.indexes = {
            'code': {'bm25': None, 'semantic': None, 'documents': []},
            'documentation': {'bm25': None, 'semantic': None, 'documents': []},
            'workflow': {'bm25': None, 'semantic': None, 'documents': []},
            'template': {'bm25': None, 'semantic': None, 'documents': []}
        }
        
        # Unified document store
        self.all_documents = []
        
        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Load processed content
        self._load_processed_content()
        
        # Try to load existing indexes
        self._load_indexes()

    def _load_processed_content(self):
        """Load processed content from content processor"""
        processed_file = self.scraped_dir / "processed_content_unified.json"
        
        if not processed_file.exists():
            logger.warning(f"Processed content file not found: {processed_file}")
            return
        
        with open(processed_file, encoding='utf-8') as f:
            data = json.load(f)
        
        content_items = data.get('content_items', [])
        logger.info(f"Loading {len(content_items)} processed content items")
        
        # Organize documents by content type
        for item in content_items:
            content_type = item['content_type']
            if content_type in self.indexes:
                self.indexes[content_type]['documents'].append(item)
                self.all_documents.append(item)
        
        # Log distribution
        for content_type, index_data in self.indexes.items():
            count = len(index_data['documents'])
            logger.info(f"  {content_type}: {count} documents")

    def build_all_indexes(self):
        """Build all indexes for multi-index architecture"""
        logger.info("🔨 Building enhanced multi-index architecture...")
        
        for content_type, index_data in self.indexes.items():
            documents = index_data['documents']
            if not documents:
                logger.info(f"  Skipping {content_type}: no documents")
                continue
                
            logger.info(f"  Building indexes for {content_type}: {len(documents)} documents")
            
            # Build BM25 index
            tokenized_docs = []
            texts_for_embedding = []
            
            for doc in documents:
                # Enhanced tokenization including code examples and API methods
                content_text = doc['content']
                if doc['code_examples']:
                    content_text += " " + " ".join(doc['code_examples'])
                if doc['api_methods']:
                    content_text += " " + " ".join(doc['api_methods'])
                if doc['tags']:
                    content_text += " " + " ".join(doc['tags'])
                    
                tokenized_docs.append(content_text.lower().split())
                
                # For semantic embedding, combine title and enhanced content
                embedding_text = f"{doc['title']} {content_text}"
                texts_for_embedding.append(embedding_text)
            
            # Create BM25 index
            index_data['bm25'] = BM25Okapi(tokenized_docs)
            
            # Create semantic index
            logger.info(f"    Encoding {len(texts_for_embedding)} texts for {content_type}...")
            embeddings = self.embedding_model.encode(texts_for_embedding, show_progress_bar=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            semantic_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            semantic_index.add(embeddings)
            
            index_data['semantic'] = semantic_index
            index_data['embeddings'] = embeddings
        
        logger.info("✅ All indexes built successfully")
        self._save_indexes()

    def search(self, 
               query: str, 
               top_k: int = 10,
               content_types: List[str] = None,
               use_case_filter: str = None,
               complexity_filter: Tuple[float, float] = None,
               source_filter: List[str] = None) -> List[EnhancedSearchResult]:
        """
        Enhanced search with filtering and content type targeting.
        
        Args:
            query: Search query
            top_k: Number of results to return
            content_types: Filter by content types ['code', 'documentation', 'workflow', 'template']
            use_case_filter: Filter by use case category
            complexity_filter: (min, max) complexity score range
            source_filter: Filter by source types
        """
        
        if not self.all_documents:
            return []
        
        # Default to all content types
        if content_types is None:
            content_types = list(self.indexes.keys())
        
        all_results = []
        
        # Search each requested content type
        for content_type in content_types:
            if content_type not in self.indexes:
                continue
                
            index_data = self.indexes[content_type]
            documents = index_data['documents']
            
            if not documents or not index_data['bm25'] or not index_data['semantic']:
                continue
            
            # Get results from this content type
            type_results = self._search_content_type(
                query, content_type, top_k * 2  # Get more to ensure diversity
            )
            
            # Apply content type weight
            content_weight = self.index_weights.get(content_type, 1.0)
            for result in type_results:
                result.score *= content_weight
            
            all_results.extend(type_results)
        
        # Apply filters
        filtered_results = self._apply_filters(
            all_results, use_case_filter, complexity_filter, source_filter
        )
        
        # Sort by score and return top k
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        return filtered_results[:top_k]

    def _search_content_type(self, query: str, content_type: str, top_k: int) -> List[EnhancedSearchResult]:
        """Search within a specific content type"""
        index_data = self.indexes[content_type]
        documents = index_data['documents']
        bm25_index = index_data['bm25']
        semantic_index = index_data['semantic']
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        
        # Semantic search
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        semantic_scores, semantic_indices = semantic_index.search(query_embedding, len(documents))
        
        # Combine scores (hybrid approach)
        combined_scores = {}
        
        # Process BM25 results
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores)
            for idx, score in enumerate(bm25_scores):
                if score > 0:
                    normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                    combined_scores[idx] = {
                        'document': documents[idx],
                        'bm25_score': normalized_score * 0.6,
                        'semantic_score': 0,
                        'total_score': normalized_score * 0.6
                    }
        
        # Process semantic results
        if len(semantic_scores[0]) > 0:
            max_semantic = max(semantic_scores[0])
            for score, idx in zip(semantic_scores[0], semantic_indices[0]):
                if idx != -1 and score > 0:
                    normalized_score = score / max_semantic if max_semantic > 0 else 0
                    
                    if idx in combined_scores:
                        combined_scores[idx]['semantic_score'] = normalized_score * 0.4
                        combined_scores[idx]['total_score'] += normalized_score * 0.4
                    else:
                        combined_scores[idx] = {
                            'document': documents[idx],
                            'bm25_score': 0,
                            'semantic_score': normalized_score * 0.4,
                            'total_score': normalized_score * 0.4
                        }
        
        # Convert to results
        results = []
        for data in combined_scores.values():
            doc = data['document']
            source_weight = self.source_weights.get(doc['source_type'], 1.0)
            
            result = EnhancedSearchResult(
                id=doc['id'],
                title=doc['title'],
                content=doc['content'],
                content_type=doc['content_type'],
                source_type=doc['source_type'],
                source_file=doc['source_file'],
                tags=doc['tags'],
                code_examples=doc['code_examples'],
                api_methods=doc['api_methods'],
                complexity_score=doc['complexity_score'],
                use_case_category=doc['use_case_category'],
                score=data['total_score'] * source_weight,
                search_source='hybrid',
                metadata=doc['metadata']
            )
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _apply_filters(self, 
                      results: List[EnhancedSearchResult],
                      use_case_filter: str = None,
                      complexity_filter: Tuple[float, float] = None,
                      source_filter: List[str] = None) -> List[EnhancedSearchResult]:
        """Apply various filters to search results"""
        
        filtered = results
        
        # Use case filter
        if use_case_filter:
            filtered = [r for r in filtered if r.use_case_category == use_case_filter]
        
        # Complexity filter
        if complexity_filter:
            min_complexity, max_complexity = complexity_filter
            filtered = [r for r in filtered 
                       if min_complexity <= r.complexity_score <= max_complexity]
        
        # Source filter
        if source_filter:
            filtered = [r for r in filtered if r.source_type in source_filter]
        
        return filtered

    def get_code_examples_for_query(self, query: str, top_k: int = 5) -> List[EnhancedSearchResult]:
        """Get code examples specifically for a query"""
        return self.search(
            query=query,
            top_k=top_k,
            content_types=['code', 'workflow'],
            source_filter=['github_python', 'github_notebook', 'ai_accelerator']
        )

    def get_documentation_for_query(self, query: str, top_k: int = 5) -> List[EnhancedSearchResult]:
        """Get documentation specifically for a query"""
        return self.search(
            query=query,
            top_k=top_k,
            content_types=['documentation'],
            source_filter=['readthedocs', 'openapi']
        )

    def get_templates_for_integration(self, platform: str, top_k: int = 3) -> List[EnhancedSearchResult]:
        """Get integration templates for specific platforms"""
        platform_query = f"{platform} integration template"
        return self.search(
            query=platform_query,
            top_k=top_k,
            content_types=['template'],
            source_filter=['ai_accelerator']
        )

    def get_use_case_examples(self, use_case: str, complexity: str = None, top_k: int = 5) -> List[EnhancedSearchResult]:
        """Get examples for specific use cases with optional complexity filtering"""
        
        complexity_ranges = {
            'simple': (0.0, 0.3),
            'intermediate': (0.3, 0.7),
            'advanced': (0.7, 1.0)
        }
        
        complexity_filter = complexity_ranges.get(complexity) if complexity else None
        
        return self.search(
            query=use_case,
            top_k=top_k,
            use_case_filter=use_case,
            complexity_filter=complexity_filter,
            content_types=['code', 'workflow', 'template']
        )

    def _save_indexes(self):
        """Save all indexes to disk"""
        logger.info("💾 Saving enhanced indexes...")
        
        try:
            for content_type, index_data in self.indexes.items():
                type_dir = self.indexes_dir / content_type
                type_dir.mkdir(exist_ok=True)
                
                # Save BM25 index
                if index_data['bm25']:
                    with open(type_dir / "bm25_index.pkl", "wb") as f:
                        pickle.dump(index_data['bm25'], f)
                
                # Save semantic index
                if index_data['semantic']:
                    faiss.write_index(index_data['semantic'], str(type_dir / "semantic_index.faiss"))
                
                # Save embeddings
                if 'embeddings' in index_data:
                    np.save(type_dir / "embeddings.npy", index_data['embeddings'])
                
                # Save documents
                with open(type_dir / "documents.json", "w", encoding='utf-8') as f:
                    json.dump(index_data['documents'], f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Enhanced indexes saved to {self.indexes_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error saving indexes: {e}")

    def _load_indexes(self):
        """Load all indexes from disk"""
        logger.info("📂 Loading enhanced indexes...")
        
        try:
            for content_type, index_data in self.indexes.items():
                type_dir = self.indexes_dir / content_type
                
                if not type_dir.exists():
                    continue
                
                # Load documents
                docs_file = type_dir / "documents.json"
                if docs_file.exists():
                    with open(docs_file, encoding='utf-8') as f:
                        index_data['documents'] = json.load(f)
                    logger.info(f"  {content_type}: {len(index_data['documents'])} documents")
                
                # Load BM25 index
                bm25_file = type_dir / "bm25_index.pkl"
                if bm25_file.exists():
                    with open(bm25_file, "rb") as f:
                        index_data['bm25'] = pickle.load(f)
                
                # Load semantic index
                semantic_file = type_dir / "semantic_index.faiss"
                if semantic_file.exists():
                    index_data['semantic'] = faiss.read_index(str(semantic_file))
                
                # Load embeddings
                embeddings_file = type_dir / "embeddings.npy"
                if embeddings_file.exists():
                    index_data['embeddings'] = np.load(embeddings_file)
            
            # Rebuild all_documents from loaded data
            self.all_documents = []
            for index_data in self.indexes.values():
                self.all_documents.extend(index_data['documents'])
            
            logger.info(f"✅ Loaded {len(self.all_documents)} total documents")
            
        except Exception as e:
            logger.error(f"❌ Error loading indexes: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive retriever statistics"""
        stats = {
            'total_documents': len(self.all_documents),
            'by_content_type': {},
            'by_source_type': {},
            'by_use_case': {},
            'complexity_distribution': {'simple': 0, 'intermediate': 0, 'advanced': 0},
            'total_code_examples': 0,
            'total_api_methods': 0,
            'index_status': {}
        }
        
        # Analyze documents
        for doc in self.all_documents:
            # Count by categories
            content_type = doc['content_type']
            source_type = doc['source_type']
            use_case = doc['use_case_category']
            
            stats['by_content_type'][content_type] = stats['by_content_type'].get(content_type, 0) + 1
            stats['by_source_type'][source_type] = stats['by_source_type'].get(source_type, 0) + 1
            stats['by_use_case'][use_case] = stats['by_use_case'].get(use_case, 0) + 1
            
            # Count examples and methods
            stats['total_code_examples'] += len(doc['code_examples'])
            stats['total_api_methods'] += len(doc['api_methods'])
            
            # Complexity distribution
            complexity = doc['complexity_score']
            if complexity < 0.3:
                stats['complexity_distribution']['simple'] += 1
            elif complexity < 0.7:
                stats['complexity_distribution']['intermediate'] += 1
            else:
                stats['complexity_distribution']['advanced'] += 1
        
        # Index status
        for content_type, index_data in self.indexes.items():
            stats['index_status'][content_type] = {
                'documents': len(index_data['documents']),
                'has_bm25': index_data['bm25'] is not None,
                'has_semantic': index_data['semantic'] is not None
            }
        
        return stats


def main():
    """Test enhanced hybrid retriever"""
    retriever = EnhancedHybridRetriever()
    
    print("🔍 Enhanced Hybrid Retriever Test")
    print("=" * 50)
    
    # Check if indexes exist, build if needed
    stats = retriever.get_stats()
    print(f"📊 Current stats: {stats['total_documents']} documents loaded")
    
    if stats['total_documents'] == 0:
        print("No documents loaded. Make sure content processor has been run.")
        return
    
    # Check if indexes need to be built
    needs_building = any(
        not index_status['has_bm25'] or not index_status['has_semantic']
        for index_status in stats['index_status'].values()
        if index_status['documents'] > 0
    )
    
    if needs_building:
        print("🔨 Building indexes (this may take a few minutes)...")
        retriever.build_all_indexes()
    
    # Test different search types
    test_queries = [
        "create a DataRobot project with time series",
        "deploy a model to production",
        "AWS integration template",
        "make batch predictions"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # General search
        results = retriever.search(query, top_k=3)
        print("  General results:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result.title} ({result.content_type}, score: {result.score:.3f})")
            print(f"       Source: {result.source_type}, Use case: {result.use_case_category}")
        
        # Code-specific search
        code_results = retriever.get_code_examples_for_query(query, top_k=2)
        if code_results:
            print("  Code examples:")
            for i, result in enumerate(code_results, 1):
                print(f"    {i}. {result.title} ({result.source_type}, {len(result.code_examples)} examples)")
    
    print(f"\n📊 Final stats:")
    final_stats = retriever.get_stats()
    for key, value in final_stats.items():
        if key != 'index_status':
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()