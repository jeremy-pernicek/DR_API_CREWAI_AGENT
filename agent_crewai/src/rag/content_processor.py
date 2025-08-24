"""
Content Processor for Unified Search Index
Preprocesses and normalizes all scraped content into a unified format optimized for code generation.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import ast
import nbformat
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ProcessedContent:
    """Unified format for all processed content"""
    id: str
    title: str
    content: str
    content_type: str  # 'code', 'documentation', 'workflow', 'template'
    source_type: str   # 'readthedocs', 'github_python', 'github_notebook', 'openapi', 'ai_accelerator'
    source_file: str
    tags: List[str]
    metadata: Dict[str, Any]
    code_examples: List[str]
    api_methods: List[str]
    complexity_score: float  # 0.0 (simple) to 1.0 (advanced)
    use_case_category: str   # 'modeling', 'deployment', 'time_series', 'integration', etc.

@dataclass 
class CodeChunk:
    """Represents a meaningful chunk of code with context"""
    code: str
    description: str
    function_name: Optional[str]
    class_name: Optional[str]
    imports: List[str]
    datarobot_methods: List[str]

class ContentProcessor:
    """Processes all scraped content into unified search-optimized format"""
    
    def __init__(self, scraped_data_dir: str = "agent_crewai/data/scraped"):
        self.scraped_data_dir = Path(scraped_data_dir)
        self.processed_content: List[ProcessedContent] = []
        
        # DataRobot API method patterns for extraction
        self.dr_patterns = [
            r'datarobot\.\w+\.\w+',  # datarobot.Project.create
            r'dr\.\w+\.\w+',         # dr.Project.create  
            r'project\.\w+',         # project.start_autopilot
            r'model\.\w+',           # model.predict
            r'deployment\.\w+',      # deployment.predict
        ]
        
        # Complexity indicators
        self.complexity_indicators = {
            'simple': ['get', 'list', 'retrieve', 'basic', 'simple', 'quick'],
            'intermediate': ['create', 'train', 'deploy', 'predict', 'workflow'],
            'advanced': ['custom', 'advanced', 'optimization', 'monitoring', 'mlops', 'integration']
        }
        
        logger.info(f"Initialized content processor for {scraped_data_dir}")

    def process_all_content(self) -> List[ProcessedContent]:
        """Process all scraped content sources into unified format"""
        logger.info("🔄 Processing all scraped content sources...")
        
        # Process each content source
        self._process_readthedocs_content()
        self._process_github_python_content()  
        self._process_github_notebooks_content()
        self._process_ai_accelerators_content()
        self._process_openapi_content()
        
        logger.info(f"✅ Processed {len(self.processed_content)} content items")
        return self.processed_content

    def _process_readthedocs_content(self):
        """Process ReadTheDocs scraped documentation"""
        readthedocs_file = self.scraped_data_dir / "datarobot_python_sdk_docs.json"
        
        if not readthedocs_file.exists():
            logger.warning("ReadTheDocs content file not found")
            return
            
        with open(readthedocs_file) as f:
            docs_data = json.load(f)
        
        logger.info(f"Processing {len(docs_data)} ReadTheDocs documents...")
        
        for doc in docs_data:
            # Extract code examples
            code_examples = self._extract_code_from_content(doc['content'])
            api_methods = self._extract_datarobot_methods(doc['content'])
            
            # Determine content type and category
            content_type = self._classify_content_type(doc['content'], doc['page_type'])
            use_case_category = self._classify_use_case(doc['url'], doc['title'], doc['content'])
            complexity_score = self._calculate_complexity_score(doc['content'], code_examples)
            
            processed = ProcessedContent(
                id=f"readthedocs_{hash(doc['url'])}",
                title=doc['title'].replace('¶', '').strip(),
                content=doc['content'],
                content_type=content_type,
                source_type='readthedocs',
                source_file=doc['url'],
                tags=self._extract_tags_from_doc(doc),
                metadata={
                    'page_type': doc['page_type'],
                    'url': doc['url'],
                    'content_length': len(doc['content'])
                },
                code_examples=code_examples,
                api_methods=api_methods,
                complexity_score=complexity_score,
                use_case_category=use_case_category
            )
            
            self.processed_content.append(processed)
    
    def _process_github_python_content(self):
        """Process GitHub Python examples"""
        github_file = self.scraped_data_dir / "github_all_comprehensive_final.json"
        
        if not github_file.exists():
            logger.warning("GitHub comprehensive content file not found")
            return
            
        with open(github_file) as f:
            github_data = json.load(f)
        
        python_files = github_data.get('python_files', [])
        logger.info(f"Processing {len(python_files)} GitHub Python files...")
        
        for py_file in python_files:
            # Parse Python code for structure
            code_chunks = self._parse_python_code(py_file['content'])
            api_methods = self._extract_datarobot_methods(py_file['content'])
            
            # Enhanced tags from file analysis
            enhanced_tags = py_file.get('tags', []) + self._analyze_python_file_tags(py_file['content'])
            
            processed = ProcessedContent(
                id=f"github_python_{hash(py_file['file_path'])}",
                title=py_file['file_name'],
                content=py_file['content'],
                content_type='code',
                source_type='github_python',
                source_file=py_file['file_path'],
                tags=list(set(enhanced_tags)),
                metadata={
                    'repo_name': py_file['repo_name'],
                    'file_path': py_file['file_path'],
                    'size': py_file['size'],
                    'code_chunks': len(code_chunks)
                },
                code_examples=[chunk.code for chunk in code_chunks],
                api_methods=api_methods,
                complexity_score=self._calculate_complexity_score(py_file['content'], [py_file['content']]),
                use_case_category=self._classify_use_case_from_path(py_file['file_path'])
            )
            
            self.processed_content.append(processed)

    def _process_github_notebooks_content(self):
        """Process GitHub Jupyter notebooks"""
        github_file = self.scraped_data_dir / "github_all_comprehensive_final.json"
        
        if not github_file.exists():
            return
            
        with open(github_file) as f:
            github_data = json.load(f)
        
        notebooks = github_data.get('notebooks', [])
        logger.info(f"Processing {len(notebooks)} GitHub notebooks...")
        
        for notebook in notebooks:
            # Extract meaningful content from notebook
            extracted_content = self._extract_notebook_content(notebook['content'])
            code_examples = extracted_content['code_cells']
            api_methods = self._extract_datarobot_methods(notebook['content'])
            
            processed = ProcessedContent(
                id=f"github_notebook_{hash(notebook['file_path'])}",
                title=notebook['file_name'].replace('.ipynb', ''),
                content=extracted_content['full_content'],
                content_type='workflow',
                source_type='github_notebook',
                source_file=notebook['file_path'],
                tags=list(set(notebook.get('tags', []) + ['jupyter-notebook', 'tutorial'])),
                metadata={
                    'repo_name': notebook['repo_name'],
                    'file_path': notebook['file_path'],
                    'size': notebook['size'],
                    'cell_count': extracted_content['cell_count'],
                    'code_cell_count': len(code_examples)
                },
                code_examples=code_examples,
                api_methods=api_methods,
                complexity_score=self._calculate_complexity_score(extracted_content['full_content'], code_examples),
                use_case_category=self._classify_use_case_from_notebook(notebook)
            )
            
            self.processed_content.append(processed)

    def _process_ai_accelerators_content(self):
        """Process AI Accelerators content"""
        ai_acc_file = self.scraped_data_dir / "ai_accelerators_direct_all.json"
        
        if not ai_acc_file.exists():
            logger.warning("AI Accelerators content file not found")
            return
            
        with open(ai_acc_file) as f:
            ai_acc_data = json.load(f)
        
        logger.info(f"Processing {len(ai_acc_data)} AI Accelerators files...")
        
        for acc_file in ai_acc_data:
            if acc_file['file_type'] == 'notebook':
                # Process notebook content
                extracted_content = self._extract_notebook_content(acc_file['content'])
                code_examples = extracted_content['code_cells']
                content = extracted_content['full_content']
            else:
                # Process other file types
                code_examples = self._extract_code_from_content(acc_file['content'])
                content = acc_file['content']
                
            api_methods = self._extract_datarobot_methods(content)
            
            processed = ProcessedContent(
                id=f"ai_accelerator_{hash(acc_file['file_path'])}",
                title=f"{acc_file['subsection']}: {acc_file['file_name']}",
                content=content,
                content_type='template',
                source_type='ai_accelerator',
                source_file=acc_file['file_path'],
                tags=acc_file.get('tags', []) + ['ai-accelerator', acc_file['section']],
                metadata={
                    'section': acc_file['section'],
                    'subsection': acc_file['subsection'],
                    'file_type': acc_file['file_type'],
                    'size': acc_file['size']
                },
                code_examples=code_examples,
                api_methods=api_methods,
                complexity_score=self._calculate_complexity_score(content, code_examples),
                use_case_category=self._classify_ai_accelerator_category(acc_file['section'], acc_file['subsection'])
            )
            
            self.processed_content.append(processed)

    def _process_openapi_content(self):
        """Process OpenAPI documentation and examples"""
        openapi_docs_file = self.scraped_data_dir / "openapi_llm_docs.json"
        openapi_examples_file = self.scraped_data_dir / "openapi_code_examples.json"
        
        if not openapi_docs_file.exists() or not openapi_examples_file.exists():
            logger.warning("OpenAPI content files not found")
            return
            
        with open(openapi_examples_file) as f:
            examples_data = json.load(f)
        
        logger.info(f"Processing {len(examples_data)} OpenAPI code examples...")
        
        for example in examples_data:
            processed = ProcessedContent(
                id=f"openapi_{hash(example['endpoint'])}",
                title=f"OpenAPI: {example['summary']}",
                content=example['code'],
                content_type='code',
                source_type='openapi',
                source_file=example['endpoint'],
                tags=['openai-api', 'api-reference', 'code-example'],
                metadata={
                    'endpoint': example['endpoint'],
                    'summary': example['summary'],
                    'language': example['language']
                },
                code_examples=[example['code']],
                api_methods=self._extract_openai_methods(example['code']),
                complexity_score=0.3,  # API examples are generally intermediate
                use_case_category='api_integration'
            )
            
            self.processed_content.append(processed)

    def _extract_code_from_content(self, content: str) -> List[str]:
        """Extract code blocks from markdown or text content"""
        code_blocks = []
        
        # Python code block patterns
        patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'.. code-block:: python\n\n(.*?)\n\n',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            code_blocks.extend(matches)
        
        # Clean and filter meaningful code blocks
        cleaned_blocks = []
        for block in code_blocks:
            block = block.strip()
            if len(block) > 20 and ('datarobot' in block.lower() or 'import' in block):
                cleaned_blocks.append(block)
        
        return cleaned_blocks

    def _extract_datarobot_methods(self, content: str) -> List[str]:
        """Extract DataRobot API method calls from content"""
        methods = []
        content_lower = content.lower()
        
        for pattern in self.dr_patterns:
            matches = re.findall(pattern, content_lower)
            methods.extend(matches)
        
        return list(set(methods))

    def _extract_openai_methods(self, content: str) -> List[str]:
        """Extract OpenAI API method calls from content"""
        methods = []
        openai_patterns = [
            r'client\.\w+\.\w+',
            r'openai\.\w+\.\w+',
            r'completions\.create',
            r'embeddings\.create',
        ]
        
        content_lower = content.lower()
        for pattern in openai_patterns:
            matches = re.findall(pattern, content_lower)
            methods.extend(matches)
            
        return list(set(methods))

    def _parse_python_code(self, code: str) -> List[CodeChunk]:
        """Parse Python code into meaningful chunks"""
        chunks = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_code = ast.get_source_segment(code, node) or ""
                    docstring = ast.get_docstring(node) or ""
                    
                    # Extract imports at module level
                    imports = []
                    for n in tree.body:
                        if isinstance(n, (ast.Import, ast.ImportFrom)):
                            imports.append(ast.get_source_segment(code, n) or "")
                    
                    # Extract DataRobot method calls
                    dr_methods = self._extract_datarobot_methods(func_code)
                    
                    chunks.append(CodeChunk(
                        code=func_code,
                        description=docstring,
                        function_name=node.name,
                        class_name=None,
                        imports=imports,
                        datarobot_methods=dr_methods
                    ))
                    
                elif isinstance(node, ast.ClassDef):
                    class_code = ast.get_source_segment(code, node) or ""
                    docstring = ast.get_docstring(node) or ""
                    
                    chunks.append(CodeChunk(
                        code=class_code,
                        description=docstring,
                        function_name=None,
                        class_name=node.name,
                        imports=[],
                        datarobot_methods=self._extract_datarobot_methods(class_code)
                    ))
                    
        except SyntaxError:
            # If parsing fails, treat entire code as one chunk
            chunks.append(CodeChunk(
                code=code,
                description="",
                function_name=None,
                class_name=None,
                imports=[],
                datarobot_methods=self._extract_datarobot_methods(code)
            ))
        
        return chunks

    def _extract_notebook_content(self, notebook_content: str) -> Dict[str, Any]:
        """Extract meaningful content from Jupyter notebook JSON"""
        try:
            nb_data = json.loads(notebook_content)
            
            code_cells = []
            markdown_cells = []
            full_content_parts = []
            
            for cell in nb_data.get('cells', []):
                cell_type = cell.get('cell_type', '')
                source = cell.get('source', [])
                
                if isinstance(source, list):
                    source_text = ''.join(source)
                else:
                    source_text = str(source)
                
                if cell_type == 'code' and source_text.strip():
                    code_cells.append(source_text)
                    full_content_parts.append(f"```python\n{source_text}\n```")
                    
                elif cell_type == 'markdown' and source_text.strip():
                    markdown_cells.append(source_text)
                    full_content_parts.append(source_text)
            
            return {
                'code_cells': code_cells,
                'markdown_cells': markdown_cells,
                'full_content': '\n\n'.join(full_content_parts),
                'cell_count': len(nb_data.get('cells', []))
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse notebook content: {str(e)}")
            return {
                'code_cells': [],
                'markdown_cells': [],
                'full_content': notebook_content[:1000],  # Fallback to raw content preview
                'cell_count': 0
            }

    def _calculate_complexity_score(self, content: str, code_examples: List[str]) -> float:
        """Calculate complexity score based on content analysis"""
        content_lower = content.lower()
        score = 0.0
        
        # Base score from code complexity
        total_code_length = sum(len(code) for code in code_examples)
        if total_code_length > 0:
            score += min(total_code_length / 1000, 0.3)  # Length factor
        
        # Keyword-based complexity scoring
        for level, keywords in self.complexity_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            if level == 'simple':
                score += matches * 0.05
            elif level == 'intermediate':  
                score += matches * 0.15
            elif level == 'advanced':
                score += matches * 0.25
        
        # API method complexity
        api_method_count = len(self._extract_datarobot_methods(content))
        score += min(api_method_count * 0.1, 0.3)
        
        return min(score, 1.0)  # Cap at 1.0

    def _classify_content_type(self, content: str, page_type: str = '') -> str:
        """Classify content into type categories"""
        content_lower = content.lower()
        
        if 'def ' in content or 'class ' in content or 'import' in content:
            return 'code'
        elif 'tutorial' in page_type or 'guide' in page_type:
            return 'workflow' 
        elif 'template' in content_lower or 'example' in content_lower:
            return 'template'
        else:
            return 'documentation'

    def _classify_use_case(self, url: str, title: str, content: str) -> str:
        """Classify content by use case category"""
        combined_text = f"{url} {title} {content}".lower()
        
        categories = {
            'modeling': ['model', 'autopilot', 'train', 'blueprint', 'algorithm'],
            'deployment': ['deploy', 'prediction', 'scoring', 'endpoint'],
            'time_series': ['time series', 'forecast', 'timeseries', 'ts_'],
            'integration': ['aws', 'azure', 'gcp', 'databricks', 'snowflake'],
            'mlops': ['mlops', 'monitoring', 'drift', 'governance'],
            'data_prep': ['feature', 'dataprep', 'preprocessing', 'engineering'],
            'genai': ['llm', 'generative', 'openai', 'genai', 'rag']
        }
        
        for category, keywords in categories.items():
            if any(keyword in combined_text for keyword in keywords):
                return category
                
        return 'general'

    def _classify_use_case_from_path(self, file_path: str) -> str:
        """Classify use case from file path"""
        path_lower = file_path.lower()
        
        if 'time' in path_lower or 'ts_' in path_lower:
            return 'time_series'
        elif 'deploy' in path_lower or 'prediction' in path_lower:
            return 'deployment'
        elif 'model' in path_lower:
            return 'modeling'
        elif 'dataprep' in path_lower:
            return 'data_prep'
        else:
            return 'general'

    def _classify_use_case_from_notebook(self, notebook: Dict[str, Any]) -> str:
        """Classify use case from notebook metadata"""
        title = notebook.get('file_name', '').lower()
        repo = notebook.get('repo_name', '').lower()
        
        if 'forecast' in title or 'forecast' in repo:
            return 'time_series'
        elif 'classification' in title:
            return 'modeling'
        elif 'deployment' in title or 'deploy' in title:
            return 'deployment'
        elif 'rag' in title or 'llm' in title:
            return 'genai'
        else:
            return 'general'

    def _classify_ai_accelerator_category(self, section: str, subsection: str) -> str:
        """Classify AI Accelerator category"""
        if section == 'ecosystem_integration_templates':
            return 'integration'
        elif section == 'generative_ai':
            return 'genai'
        elif section == 'use_cases_and_horizontal_approaches':
            return 'business_use_case'
        else:
            return 'general'

    def _extract_tags_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        """Extract tags from document content"""
        tags = []
        
        # Add existing tags
        if 'tags' in doc:
            tags.extend(doc['tags'])
            
        # Add page type
        if 'page_type' in doc:
            tags.append(doc['page_type'])
        
        # Content-based tags
        content_lower = doc['content'].lower()
        
        tag_keywords = {
            'api-reference': ['api reference', 'endpoint', 'parameter'],
            'tutorial': ['tutorial', 'step by step', 'walkthrough'],
            'example': ['example', 'sample', 'demo'],
            'advanced': ['advanced', 'custom', 'optimization'],
            'beginner': ['basic', 'simple', 'introduction', 'getting started']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tag)
        
        return list(set(tags))

    def _analyze_python_file_tags(self, content: str) -> List[str]:
        """Analyze Python file content for additional tags"""
        tags = []
        content_lower = content.lower()
        
        # Framework/library tags
        if 'streamlit' in content_lower:
            tags.append('streamlit')
        if 'fastapi' in content_lower:
            tags.append('fastapi')
        if 'pandas' in content_lower:
            tags.append('pandas')
        if 'numpy' in content_lower:
            tags.append('numpy')
        
        # DataRobot specific patterns
        if 'autopilot' in content_lower:
            tags.append('autopilot')
        if 'blueprint' in content_lower:
            tags.append('blueprint')
        if 'feature_list' in content_lower:
            tags.append('feature-engineering')
        
        return tags

    def save_processed_content(self, output_file: str = "processed_content_unified.json"):
        """Save processed content to file"""
        output_path = self.scraped_data_dir / output_file
        
        # Convert to serializable format
        serializable_content = [asdict(item) for item in self.processed_content]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'processed_at': datetime.now().isoformat(),
                'total_items': len(serializable_content),
                'content_items': serializable_content
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(self.processed_content)} processed content items to {output_path}")
        
        # Create summary
        self._create_processing_summary()

    def _create_processing_summary(self):
        """Create processing summary statistics"""
        summary = {
            'total_items': len(self.processed_content),
            'by_content_type': {},
            'by_source_type': {},
            'by_use_case_category': {},
            'complexity_distribution': {},
            'top_tags': {},
            'total_code_examples': 0,
            'total_api_methods': 0
        }
        
        all_tags = []
        complexity_ranges = {'simple': 0, 'intermediate': 0, 'advanced': 0}
        
        for item in self.processed_content:
            # Count by categories
            summary['by_content_type'][item.content_type] = summary['by_content_type'].get(item.content_type, 0) + 1
            summary['by_source_type'][item.source_type] = summary['by_source_type'].get(item.source_type, 0) + 1
            summary['by_use_case_category'][item.use_case_category] = summary['by_use_case_category'].get(item.use_case_category, 0) + 1
            
            # Collect tags
            all_tags.extend(item.tags)
            
            # Count examples and methods
            summary['total_code_examples'] += len(item.code_examples)
            summary['total_api_methods'] += len(item.api_methods)
            
            # Complexity distribution
            if item.complexity_score < 0.3:
                complexity_ranges['simple'] += 1
            elif item.complexity_score < 0.7:
                complexity_ranges['intermediate'] += 1
            else:
                complexity_ranges['advanced'] += 1
        
        # Top tags
        from collections import Counter
        tag_counts = Counter(all_tags)
        summary['top_tags'] = dict(tag_counts.most_common(20))
        summary['complexity_distribution'] = complexity_ranges
        
        # Save summary
        summary_path = self.scraped_data_dir / "content_processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Processing summary saved to {summary_path}")


def main():
    """Test content processor"""
    processor = ContentProcessor()
    
    print("🔄 Processing all scraped content...")
    processed_items = processor.process_all_content()
    
    print(f"\n📊 Processing Results:")
    print(f"   Total items: {len(processed_items)}")
    
    # Show distribution
    content_types = {}
    source_types = {}
    
    for item in processed_items:
        content_types[item.content_type] = content_types.get(item.content_type, 0) + 1
        source_types[item.source_type] = source_types.get(item.source_type, 0) + 1
    
    print(f"\n📄 By Content Type:")
    for ctype, count in content_types.items():
        print(f"   {ctype}: {count}")
    
    print(f"\n📚 By Source Type:")
    for stype, count in source_types.items():
        print(f"   {stype}: {count}")
    
    # Save results
    processor.save_processed_content()
    
    print(f"\n✅ Content processing complete!")

if __name__ == "__main__":
    main()