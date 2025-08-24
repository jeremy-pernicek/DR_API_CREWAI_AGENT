"""
Smart Context Assembly Pipeline
Intelligently assembles context from search results for optimal code generation.
"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .enhanced_hybrid_retriever import EnhancedHybridRetriever, EnhancedSearchResult

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries for different prompt strategies"""
    HOW_TO = "how_to"           # "How do I create a project?"
    DEBUG = "debug"             # "Fix this code that's not working"
    CREATE_WORKFLOW = "workflow" # "Create a workflow for time series forecasting"
    API_REFERENCE = "api_ref"   # "What parameters does Project.create take?"
    EXAMPLE_REQUEST = "example" # "Show me an example of batch predictions"
    INTEGRATION = "integration" # "How do I integrate with AWS?"

@dataclass
class AssembledContext:
    """Assembled context ready for LLM consumption"""
    primary_context: str
    code_examples: List[str]
    api_references: List[str]
    related_workflows: List[str]
    complexity_level: str  # 'beginner', 'intermediate', 'advanced'
    use_case_category: str
    source_citations: List[Dict[str, str]]
    total_tokens: int  # Estimated token count
    query_type: QueryType = None
    
    @property
    def formatted_context(self) -> str:
        """Format the assembled context for LLM consumption"""
        context_parts = []
        
        if self.primary_context:
            context_parts.append(f"## Context\n{self.primary_context}")
        
        if self.code_examples:
            context_parts.append("## Code Examples")
            for i, example in enumerate(self.code_examples, 1):
                context_parts.append(f"### Example {i}\n```python\n{example}\n```")
        
        if self.api_references:
            context_parts.append("## API References")
            context_parts.extend(self.api_references)
        
        if self.related_workflows:
            context_parts.append("## Related Workflows")
            context_parts.extend(self.related_workflows)
        
        return "\n\n".join(context_parts)
    
    @property
    def summary(self) -> str:
        """Generate a brief summary of the assembled context"""
        parts = []
        if self.code_examples:
            parts.append(f"{len(self.code_examples)} code examples")
        if self.api_references:
            parts.append(f"{len(self.api_references)} API refs")
        if self.related_workflows:
            parts.append(f"{len(self.related_workflows)} workflows")
        
        base = f"{self.use_case_category} ({self.complexity_level})"
        if parts:
            return f"{base} with {', '.join(parts)}"
        return base
    
    @property
    def content_types(self) -> List[str]:
        """Get the content types represented in this context"""
        types = []
        if self.code_examples:
            types.append('code')
        if self.api_references:
            types.append('api_reference')
        if self.related_workflows:
            types.append('workflow')
        if self.primary_context:
            types.append('documentation')
        return types
    
    @property
    def source_types(self) -> List[str]:
        """Get the source types from citations"""
        if not self.source_citations:
            return []
        
        sources = set()
        for citation in self.source_citations:
            if 'source_type' in citation:
                sources.add(citation['source_type'])
        return list(sources)

class ContextAssembler:
    """Smart context assembly pipeline"""
    
    def __init__(self, retriever: EnhancedHybridRetriever):
        self.retriever = retriever
        
        # Token limits for different LLMs (approximate)
        self.token_limits = {
            'claude': 180000,    # Claude Sonnet 4 context limit
            'gpt-4o': 120000,    # GPT-4o context limit  
            'gemini': 100000     # Gemini 2.5 Pro context limit
        }
        
        # Context assembly templates
        self.prompt_templates = {
            QueryType.HOW_TO: self._get_how_to_template(),
            QueryType.DEBUG: self._get_debug_template(),
            QueryType.CREATE_WORKFLOW: self._get_workflow_template(),
            QueryType.API_REFERENCE: self._get_api_reference_template(),
            QueryType.EXAMPLE_REQUEST: self._get_example_template(),
            QueryType.INTEGRATION: self._get_integration_template()
        }

    def assemble_context(self, 
                        query: str,
                        model_name: str = 'claude',
                        max_examples: int = 5,
                        complexity_preference: str = None) -> AssembledContext:
        """
        Assemble optimal context for a query.
        
        Args:
            query: User's query
            model_name: Target LLM model name
            max_examples: Maximum number of code examples to include
            complexity_preference: 'simple', 'intermediate', 'advanced', or None for auto
        """
        
        # Classify query type
        query_type = self._classify_query_type(query)
        logger.info(f"Classified query as: {query_type.value}")
        
        # Get relevant content with smart filtering
        search_results = self._intelligent_search(query, query_type, complexity_preference)
        
        # Assemble context based on query type
        context = self._assemble_by_query_type(query, query_type, search_results, max_examples)
        
        # Optimize for target model's token limits
        context = self._optimize_for_model(context, model_name)
        
        logger.info(f"Assembled context: {context.total_tokens} estimated tokens")
        return context

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query into appropriate type"""
        query_lower = query.lower()
        
        # How-to patterns
        if any(pattern in query_lower for pattern in ['how to', 'how do i', 'how can i', 'steps to']):
            return QueryType.HOW_TO
        
        # Debug patterns  
        if any(pattern in query_lower for pattern in ['fix', 'error', 'not working', 'debug', 'issue', 'problem']):
            return QueryType.DEBUG
        
        # Workflow creation patterns
        if any(pattern in query_lower for pattern in ['create workflow', 'build pipeline', 'end to end', 'complete process']):
            return QueryType.CREATE_WORKFLOW
        
        # Integration patterns
        if any(pattern in query_lower for pattern in ['integrate', 'aws', 'azure', 'gcp', 'databricks', 'snowflake']):
            return QueryType.INTEGRATION
        
        # API reference patterns
        if any(pattern in query_lower for pattern in ['parameters', 'arguments', 'api reference', 'function signature']):
            return QueryType.API_REFERENCE
        
        # Example request patterns
        if any(pattern in query_lower for pattern in ['example', 'sample', 'show me', 'demonstrate']):
            return QueryType.EXAMPLE_REQUEST
        
        # Default to how-to
        return QueryType.HOW_TO

    def _intelligent_search(self, 
                           query: str, 
                           query_type: QueryType, 
                           complexity_preference: str = None) -> Dict[str, List[EnhancedSearchResult]]:
        """Perform intelligent search based on query type"""
        
        results = {
            'primary': [],
            'code_examples': [],
            'documentation': [],
            'workflows': [],
            'templates': []
        }
        
        # Primary search strategy based on query type
        if query_type == QueryType.HOW_TO:
            # Focus on documentation + working examples
            results['primary'] = self.retriever.search(
                query, top_k=5, 
                content_types=['documentation', 'code', 'workflow']
            )
            results['code_examples'] = self.retriever.get_code_examples_for_query(query, top_k=3)
            
        elif query_type == QueryType.DEBUG:
            # Focus on working code examples and common issues
            results['code_examples'] = self.retriever.get_code_examples_for_query(query, top_k=5)
            results['documentation'] = self.retriever.get_documentation_for_query(query, top_k=2)
            
        elif query_type == QueryType.CREATE_WORKFLOW:
            # Focus on complete workflows and templates
            results['workflows'] = self.retriever.search(
                query, top_k=3, 
                content_types=['workflow']
            )
            results['templates'] = self.retriever.search(
                query, top_k=3,
                content_types=['template']
            )
            results['code_examples'] = self.retriever.get_code_examples_for_query(query, top_k=2)
            
        elif query_type == QueryType.INTEGRATION:
            # Focus on integration templates and examples
            platform = self._extract_platform_from_query(query)
            if platform:
                results['templates'] = self.retriever.get_templates_for_integration(platform, top_k=3)
            results['code_examples'] = self.retriever.get_code_examples_for_query(query, top_k=3)
            results['workflows'] = self.retriever.search(
                query, top_k=2,
                content_types=['workflow'],
                source_filter=['ai_accelerator', 'github_notebook']
            )
            
        elif query_type == QueryType.API_REFERENCE:
            # Focus on API documentation and examples
            results['documentation'] = self.retriever.get_documentation_for_query(query, top_k=5)
            results['code_examples'] = self.retriever.get_code_examples_for_query(query, top_k=3)
            
        elif query_type == QueryType.EXAMPLE_REQUEST:
            # Focus on code examples and workflows
            results['code_examples'] = self.retriever.get_code_examples_for_query(query, top_k=5)
            results['workflows'] = self.retriever.search(
                query, top_k=3,
                content_types=['workflow']
            )
        
        # Apply complexity filtering if specified
        if complexity_preference:
            results = self._filter_by_complexity(results, complexity_preference)
        
        return results

    def _assemble_by_query_type(self, 
                                query: str,
                                query_type: QueryType, 
                                search_results: Dict[str, List[EnhancedSearchResult]], 
                                max_examples: int) -> AssembledContext:
        """Assemble context based on query type"""
        
        # Determine complexity level from results
        complexity_level = self._determine_complexity_level(search_results)
        
        # Determine use case category
        use_case_category = self._determine_use_case_category(search_results)
        
        # Assemble different context components
        primary_context = self._build_primary_context(query, query_type, search_results)
        code_examples = self._extract_code_examples(search_results, max_examples)
        api_references = self._extract_api_references(search_results)
        related_workflows = self._extract_workflows(search_results)
        source_citations = self._build_citations(search_results)
        
        # Estimate tokens
        total_content = f"{primary_context} {' '.join(code_examples)} {' '.join(api_references)} {' '.join(related_workflows)}"
        estimated_tokens = self._estimate_tokens(total_content)
        
        return AssembledContext(
            primary_context=primary_context,
            code_examples=code_examples,
            api_references=api_references,
            related_workflows=related_workflows,
            complexity_level=complexity_level,
            use_case_category=use_case_category,
            source_citations=source_citations,
            total_tokens=estimated_tokens,
            query_type=query_type
        )

    def _build_primary_context(self, 
                              query: str, 
                              query_type: QueryType, 
                              search_results: Dict[str, List[EnhancedSearchResult]]) -> str:
        """Build primary context section"""
        
        context_parts = []
        
        # Add query-specific introduction
        template = self.prompt_templates[query_type]
        context_parts.append(template['introduction'])
        
        # Add most relevant primary results
        primary_results = (search_results.get('primary', []) + 
                          search_results.get('documentation', []) + 
                          search_results.get('workflows', []))
        
        if primary_results:
            context_parts.append("\n## Relevant Documentation:\n")
            for i, result in enumerate(primary_results[:3], 1):
                context_parts.append(f"**{i}. {result.title}** ({result.source_type})")
                # Truncate content for context
                content_preview = result.content[:800] + "..." if len(result.content) > 800 else result.content
                context_parts.append(content_preview)
                context_parts.append("")
        
        return "\n".join(context_parts)

    def _extract_code_examples(self, 
                              search_results: Dict[str, List[EnhancedSearchResult]], 
                              max_examples: int) -> List[str]:
        """Extract and format code examples"""
        
        examples = []
        
        # Collect code examples from all result types
        all_results = []
        for result_list in search_results.values():
            all_results.extend(result_list)
        
        # Sort by relevance score and extract code
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        for result in all_results:
            if len(examples) >= max_examples:
                break
                
            # Add code examples from the result
            for code_example in result.code_examples:
                if len(examples) >= max_examples:
                    break
                    
                if len(code_example.strip()) > 50:  # Only meaningful code blocks
                    formatted_example = f"# From: {result.title} ({result.source_type})\n{code_example}"
                    examples.append(formatted_example)
        
        return examples

    def _extract_api_references(self, search_results: Dict[str, List[EnhancedSearchResult]]) -> List[str]:
        """Extract API method references"""
        
        api_refs = []
        
        # Focus on documentation and code results for API methods
        doc_results = search_results.get('documentation', []) + search_results.get('primary', [])
        
        for result in doc_results[:3]:
            if result.api_methods:
                api_info = f"**{result.title}** API Methods:\n"
                for method in result.api_methods[:5]:  # Limit per result
                    api_info += f"- {method}\n"
                api_refs.append(api_info)
        
        return api_refs

    def _extract_workflows(self, search_results: Dict[str, List[EnhancedSearchResult]]) -> List[str]:
        """Extract workflow descriptions"""
        
        workflows = []
        
        workflow_results = search_results.get('workflows', []) + search_results.get('templates', [])
        
        for result in workflow_results[:2]:
            if result.content_type in ['workflow', 'template']:
                workflow_desc = f"**Workflow: {result.title}** ({result.source_type})\n"
                # Extract first meaningful paragraph
                content_lines = result.content.split('\n')
                meaningful_content = []
                for line in content_lines[:10]:  # First 10 lines
                    line = line.strip()
                    if len(line) > 30 and not line.startswith('#') and not line.startswith('```'):
                        meaningful_content.append(line)
                        if len(meaningful_content) >= 3:  # Max 3 lines per workflow
                            break
                
                if meaningful_content:
                    workflow_desc += "\n".join(meaningful_content)
                    workflows.append(workflow_desc)
        
        return workflows

    def _build_citations(self, search_results: Dict[str, List[EnhancedSearchResult]]) -> List[Dict[str, str]]:
        """Build source citations"""
        
        citations = []
        seen_sources = set()
        
        # Collect all unique sources
        all_results = []
        for result_list in search_results.values():
            all_results.extend(result_list)
        
        for result in all_results:
            source_key = f"{result.source_type}_{result.source_file}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                citations.append({
                    'title': result.title,
                    'source_type': result.source_type,
                    'source_file': result.source_file,
                    'url': getattr(result, 'url', result.source_file)
                })
                
                if len(citations) >= 10:  # Limit citations
                    break
        
        return citations

    def _determine_complexity_level(self, search_results: Dict[str, List[EnhancedSearchResult]]) -> str:
        """Determine appropriate complexity level from results"""
        
        all_results = []
        for result_list in search_results.values():
            all_results.extend(result_list)
        
        if not all_results:
            return 'intermediate'
        
        avg_complexity = sum(r.complexity_score for r in all_results) / len(all_results)
        
        if avg_complexity < 0.3:
            return 'beginner'
        elif avg_complexity < 0.7:
            return 'intermediate'
        else:
            return 'advanced'

    def _determine_use_case_category(self, search_results: Dict[str, List[EnhancedSearchResult]]) -> str:
        """Determine primary use case category"""
        
        categories = {}
        
        all_results = []
        for result_list in search_results.values():
            all_results.extend(result_list)
        
        for result in all_results:
            category = result.use_case_category
            categories[category] = categories.get(category, 0) + 1
        
        if not categories:
            return 'general'
        
        # Return most common category
        return max(categories.items(), key=lambda x: x[1])[0]

    def _filter_by_complexity(self, 
                             search_results: Dict[str, List[EnhancedSearchResult]], 
                             complexity_preference: str) -> Dict[str, List[EnhancedSearchResult]]:
        """Filter results by complexity preference"""
        
        complexity_ranges = {
            'simple': (0.0, 0.4),
            'intermediate': (0.2, 0.8),
            'advanced': (0.6, 1.0)
        }
        
        min_complexity, max_complexity = complexity_ranges.get(complexity_preference, (0.0, 1.0))
        
        filtered_results = {}
        for key, result_list in search_results.items():
            filtered_list = [
                result for result in result_list
                if min_complexity <= result.complexity_score <= max_complexity
            ]
            filtered_results[key] = filtered_list
        
        return filtered_results

    def _extract_platform_from_query(self, query: str) -> Optional[str]:
        """Extract platform name from integration query"""
        query_lower = query.lower()
        
        platforms = ['aws', 'azure', 'gcp', 'databricks', 'snowflake', 'sagemaker', 'athena', 's3']
        
        for platform in platforms:
            if platform in query_lower:
                return platform
        
        return None

    def _optimize_for_model(self, context: AssembledContext, model_name: str) -> AssembledContext:
        """Optimize context for specific model token limits"""
        
        token_limit = self.token_limits.get(model_name, 100000)
        
        if context.total_tokens <= token_limit:
            return context
        
        # Reduce context size by priority
        # 1. Trim workflows first
        if context.total_tokens > token_limit and len(context.related_workflows) > 1:
            context.related_workflows = context.related_workflows[:1]
        
        # 2. Reduce code examples
        if context.total_tokens > token_limit and len(context.code_examples) > 2:
            context.code_examples = context.code_examples[:2]
        
        # 3. Trim primary context
        if context.total_tokens > token_limit:
            context.primary_context = context.primary_context[:int(len(context.primary_context) * 0.7)]
        
        # Recalculate tokens
        total_content = f"{context.primary_context} {' '.join(context.code_examples)} {' '.join(context.api_references)} {' '.join(context.related_workflows)}"
        context.total_tokens = self._estimate_tokens(total_content)
        
        return context

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (approximately 4 chars per token)"""
        return len(text) // 4

    def _get_how_to_template(self) -> Dict[str, str]:
        """Template for how-to queries"""
        return {
            'introduction': """I'll help you accomplish this task with DataRobot. Here's the relevant information and working examples:"""
        }

    def _get_debug_template(self) -> Dict[str, str]:
        """Template for debugging queries"""
        return {
            'introduction': """I'll help you debug this issue. Here are working examples and common solutions:"""
        }

    def _get_workflow_template(self) -> Dict[str, str]:
        """Template for workflow creation queries"""
        return {
            'introduction': """I'll help you create a complete workflow. Here are proven patterns and templates:"""
        }

    def _get_api_reference_template(self) -> Dict[str, str]:
        """Template for API reference queries"""
        return {
            'introduction': """Here's the API reference information you requested:"""
        }

    def _get_example_template(self) -> Dict[str, str]:
        """Template for example requests"""
        return {
            'introduction': """Here are working examples that demonstrate what you're looking for:"""
        }

    def _get_integration_template(self) -> Dict[str, str]:
        """Template for integration queries"""
        return {
            'introduction': """Here are integration templates and examples for your platform:"""
        }


def main():
    """Test context assembler"""
    from enhanced_hybrid_retriever import EnhancedHybridRetriever
    
    print("🔗 Context Assembler Test")
    print("=" * 40)
    
    # Initialize retriever and context assembler
    retriever = EnhancedHybridRetriever()
    assembler = ContextAssembler(retriever)
    
    if retriever.get_stats()['total_documents'] == 0:
        print("No documents loaded. Please run the enhanced hybrid retriever first.")
        return
    
    # Test different query types
    test_queries = [
        "How do I create a DataRobot time series project?",
        "Show me an example of batch predictions",
        "Fix my model deployment code that's not working",
        "Create a workflow for AWS integration",
        "What parameters does Project.start take?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        try:
            context = assembler.assemble_context(query, model_name='claude', max_examples=3)
            
            print(f"  📊 Context Stats:")
            print(f"    Complexity: {context.complexity_level}")
            print(f"    Use case: {context.use_case_category}")
            print(f"    Estimated tokens: {context.total_tokens}")
            print(f"    Code examples: {len(context.code_examples)}")
            print(f"    API references: {len(context.api_references)}")
            print(f"    Workflows: {len(context.related_workflows)}")
            print(f"    Citations: {len(context.source_citations)}")
            
            # Show snippet of primary context
            if context.primary_context:
                preview = context.primary_context[:200] + "..." if len(context.primary_context) > 200 else context.primary_context
                print(f"  📝 Context preview: {preview}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

if __name__ == "__main__":
    main()