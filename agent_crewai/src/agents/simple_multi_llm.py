"""
Simple multi-LLM orchestration for DataRobot API code generation.
Alternative to CrewAI that works with Python 3.9.
"""

import os
import sys
import time
import datarobot as dr
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.enhanced_hybrid_retriever import EnhancedHybridRetriever
from rag.context_assembler import ContextAssembler, AssembledContext
from agents.consolidated_prompts import CONSOLIDATED_PROMPTS
from agents.practical_prompts import PRACTICAL_PROMPTS
from agents.shared_models import CodeSolution, GenerationMetrics, SynthesisStrategy
from agents.code_synthesizer import DataRobotCodeSynthesizer
from agents.synthesis_orchestrator import SynthesisOrchestrator
from agents.component_extractors import ComponentExtractionOrchestrator

load_dotenv()

class SimpleMultiLLM:
    """Enhanced multi-LLM orchestration with unified search system"""
    
    def __init__(self, retriever=None, data_dir: str = "agent_crewai/data", init_dr_client: bool = True, use_practical_prompts: bool = True):
        """Initialize the enhanced multi-LLM system."""
        
        # Initialize DataRobot client with error handling
        self.dr_client = None
        self.llm_base_url = None
        self.dr_api_token = os.getenv('DATAROBOT_API_TOKEN')
        self.use_practical_prompts = use_practical_prompts
        self.prompts = PRACTICAL_PROMPTS if use_practical_prompts else CONSOLIDATED_PROMPTS
        
        if init_dr_client and self.dr_api_token:
            try:
                # Clean the API token of any potential Unicode issues
                clean_token = self.dr_api_token.encode('ascii', errors='ignore').decode('ascii')
                
                # Set up environment with clean token
                os.environ['DATAROBOT_API_TOKEN'] = clean_token
                
                self.dr_client = dr.Client()
                self.llm_base_url = f"{self.dr_client.endpoint}/genai/llmgw"
                
            except Exception as e:
                print(f"⚠️  DataRobot client initialization failed: {e}")
                print(f"   Continuing with retrieval-only mode...")
                
                # Fall back to default endpoint for LLM calls
                endpoint = os.getenv('DATAROBOT_ENDPOINT', 'https://app.datarobot.com/api/v2')
                self.llm_base_url = f"{endpoint}/genai/llmgw"
        else:
            print(f"🔧 Running in test mode without DataRobot client initialization")
            # Fall back to default endpoint for LLM calls  
            endpoint = os.getenv('DATAROBOT_ENDPOINT', 'https://app.datarobot.com/api/v2')
            self.llm_base_url = f"{endpoint}/genai/llmgw"
        
        # Initialize enhanced retrieval system
        if retriever is None:
            print("📚 Searching DataRobot documentation, API references, and community examples...")
            self.retriever = EnhancedHybridRetriever(data_dir=data_dir)
            stats = self.retriever.get_stats()
            print(f"   Found {stats['total_documents']} relevant resources")
        else:
            self.retriever = retriever
        
        # Initialize context assembler
        self.context_assembler = ContextAssembler(self.retriever)
        
        # Initialize synthesis system
        self.code_synthesizer = DataRobotCodeSynthesizer()
        self.synthesis_orchestrator = SynthesisOrchestrator(self.code_synthesizer, verbose=True)
        self.synthesis_orchestrator.set_quiet_mode(True)  # Use concise output for production
        self.component_extractor = ComponentExtractionOrchestrator()
        
        # Updated models with working Gemini 2.5 Pro
        self.models = {
            'analyzer': 'vertex_ai/gemini-2.5-pro',
            'claude': 'bedrock/anthropic.claude-opus-4-1-20250805-v1:0',
            'gpt': 'azure/gpt-4o-2024-11-20',
            'gemini': 'vertex_ai/gemini-2.5-pro',
            'scorer': 'vertex_ai/gemini-2.5-pro'
        }
        
        # Initialize OpenAI client for DataRobot Gateway
        if self.llm_base_url and self.dr_api_token:
            try:
                self.client = OpenAI(
                    base_url=self.llm_base_url,
                    api_key=clean_token if 'clean_token' in locals() else self.dr_api_token
                )
            except Exception as e:
                print(f"⚠️  LLM Gateway client initialization failed: {e}")
                self.client = None
        else:
            print(f"🔧 LLM Gateway client not available - API calls will be skipped")
            self.client = None
    
    def _print_progress(self, message: str, stage: str = "INFO", indent: int = 0):
        """Enhanced progress printing with consistent formatting"""
        stage_emojis = {
            "INIT": "🚀",
            "CONTEXT": "📚", 
            "ANALYSIS": "🔍",
            "GENERATE": "🤖",
            "SCORE": "📊", 
            "SYNTHESIZE": "🧬",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "TIMING": "⏱️",
            "INFO": "ℹ️"
        }
        
        emoji = stage_emojis.get(stage, "•")
        indent_str = "  " * indent
        print(f"{indent_str}{emoji} {message}")
    
    def _measure_time(self, func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        return result, elapsed
    
    def _format_time(self, seconds: float) -> str:
        """Format time as minutes:seconds or just seconds if under 60s"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.2f}s"
    
    def _get_model_display_name(self, model_key: str) -> str:
        """Get human-readable display name for model"""
        model_names = {
            'analyzer': 'Gemini 2.5 Pro (Requirements Analysis)',
            'claude': 'Claude Sonnet 4 (Production-Ready)',
            'gpt': 'GPT-4o (Enterprise-Grade)', 
            'gemini': 'Gemini 2.5 Pro (Analytical)',
            'scorer': 'Gemini 2.5 Pro (Code Evaluation)'
        }
        return model_names.get(model_key, f"{model_key.title()}")
    
    def _get_model_technical_name(self, model_key: str) -> str:
        """Get technical model name for detailed output"""
        return self.models.get(model_key, f"unknown-{model_key}")
        
    def _call_llm(self, model: str, messages: List[Dict], temperature: float = 0.1) -> str:
        """Make a call to LLM via DataRobot Gateway"""
        
        # Check if client is available
        if not self.client:
            return f"""# DEMO MODE - LLM API UNAVAILABLE
# This would be generated by {model}
# 
# Sample DataRobot SDK Code:
import datarobot as dr
import os

def create_project():
    client = dr.Client(token=os.getenv('DATAROBOT_API_TOKEN'))
    project = dr.Project.start(
        dataset='data.csv',
        target='target_column'
    )
    return project

if __name__ == "__main__":
    project = create_project()
    print(f"Project created: {{project.id}}")
"""
        
        try:
            # Special handling for Gemini models
            if 'gemini' in model.lower():
                # Convert system + user messages to single user message
                if len(messages) > 1 and messages[0]['role'] == 'system':
                    system_content = messages[0]['content']
                    user_content = messages[1]['content']
                    combined_content = f"{system_content}\\n\\n{user_content}"
                    
                    modified_messages = [{"role": "user", "content": combined_content}]
                else:
                    modified_messages = messages
                
                # Gemini fails with extra_body parameters, use minimal params
                response = self.client.chat.completions.create(
                    model=model,
                    messages=modified_messages,
                    temperature=temperature
                )
            else:
                # Other models can use extra_body
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    extra_body={"max_completion_tokens": 2000}
                )
            
            content = response.choices[0].message.content
            
            # Handle case where model returns None or empty
            if content is None or len(content) == 0:
                return f"Empty response from {model}"
            
            return content
            
        except Exception as e:
            return f"Error calling {model}: {str(e)}"
    
    def _analyze_requirements(self, user_request: str) -> str:
        """Analyze user requirements using consolidated prompts with verifiable grounding"""
        
        # Use context assembler for intelligent context gathering
        assembled_context = self.context_assembler.assemble_context(
            query=user_request,
            model_name='analyzer',
            max_examples=8,
            complexity_preference='advanced'
        )
        
        # Build context pack with source IDs for reference mapping
        context_pack_with_ids = self._build_context_pack_with_ids(assembled_context)
        
        messages = [
            {
                "role": "system",
                "content": self.prompts['analyzer']
            },
            {
                "role": "user",
                "content": f"""TASK_SPEC: {user_request}

CONTEXT_PACK:
{context_pack_with_ids}

Query Classification:
- Type: {assembled_context.query_type.value if hasattr(assembled_context, 'query_type') else 'general'}
- Complexity: {assembled_context.complexity_level}
- Use Case: {assembled_context.use_case_category}
- Sources: {', '.join(assembled_context.source_types)}

Provide your analysis in the exact YAML format specified. Ensure every DataRobot SDK component is traced to a source ID in the REFERENCE_MAP."""
            }
        ]
        
        return self._call_llm(self.models['analyzer'], messages, temperature=0.0)
    
    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from LLM response, removing YAML reference mapping."""
        import re
        
        # Step 1: Remove YAML reference map section if present
        yaml_pattern = r'```yaml\s*\nREFERENCE_MAP_CONFIRMATION:.*?```\s*\n'
        clean_response = re.sub(yaml_pattern, '', response, flags=re.DOTALL)
        
        # Step 2: Check if entire response is wrapped in a code block
        full_block_pattern = r'^```python\s*\n(.*?)```\s*$'
        full_block_match = re.match(full_block_pattern, clean_response, re.DOTALL)
        
        if full_block_match:
            # Entire response is one code block - extract content
            extracted = full_block_match.group(1).strip()
        else:
            # Step 3: Extract all Python code blocks
            code_blocks = re.findall(r'```python\s*\n(.*?)```', clean_response, re.DOTALL)
            
            if code_blocks:
                # Join all code blocks
                extracted = '\n\n'.join(code_blocks).strip()
            else:
                # Step 4: Check if it's already clean code (no fences)
                if '#!' in clean_response and '```' not in clean_response:
                    extracted = clean_response.strip()
                else:
                    # Step 5: Last resort - remove any remaining fence markers
                    extracted = clean_response.replace('```python', '').replace('```', '').strip()
        
        # Step 6: Final cleanup - remove any stray fence markers
        extracted = extracted.replace('```python', '').replace('```', '').strip()
        
        # Step 7: Validate and fix common issues  
        if extracted.startswith('python'):
            # Remove stray 'python' at the beginning
            extracted = extracted[6:].strip()
        
        return extracted
    
    def _build_context_pack_with_ids(self, assembled_context: AssembledContext) -> str:
        """Build context pack with source IDs for reference mapping"""
        context_parts = []
        
        # Add context with source tracking
        if assembled_context.primary_context:
            context_parts.append(f"[CONTEXT_ID: primary_context]")
            context_parts.append(assembled_context.primary_context)
        
        # Add code examples with IDs
        if assembled_context.code_examples:
            context_parts.append("\n[CODE_EXAMPLES]")
            for i, example in enumerate(assembled_context.code_examples):
                context_parts.append(f"[EXAMPLE_ID: code_example_{i+1}]")
                context_parts.append(f"```python\n{example}\n```")
        
        # Add API references with IDs
        if assembled_context.api_references:
            context_parts.append("\n[API_REFERENCES]")
            for i, ref in enumerate(assembled_context.api_references):
                context_parts.append(f"[API_REF_ID: api_ref_{i+1}]")
                context_parts.append(ref)
        
        # Add workflows with IDs
        if assembled_context.related_workflows:
            context_parts.append("\n[WORKFLOWS]")
            for i, workflow in enumerate(assembled_context.related_workflows):
                context_parts.append(f"[WORKFLOW_ID: workflow_{i+1}]")
                context_parts.append(workflow)
        
        # Add source citations for traceability
        if assembled_context.source_citations:
            context_parts.append("\n[SOURCE_CITATIONS]")
            for citation in assembled_context.source_citations:
                source_id = citation.get('id', 'unknown')
                source_type = citation.get('source_type', 'unknown')
                context_parts.append(f"[SOURCE_ID: {source_id}] - Type: {source_type}")
        
        return "\n\n".join(context_parts)
    
    def _generate_code_consolidated(self, user_request: str, analysis: str, model_type: str = 'claude') -> CodeSolution:
        """Generate code using consolidated prompts with strict grounding enforcement"""
        
        # Assemble context for code generation
        assembled_context = self.context_assembler.assemble_context(
            query=user_request,
            model_name=model_type,
            max_examples=8,
            complexity_preference='advanced'
        )
        
        # Build context pack with source IDs
        context_pack_with_ids = self._build_context_pack_with_ids(assembled_context)
        
        # Select the appropriate prompt based on model type
        if model_type == 'gemini':
            prompt_key = 'gemini'
        else:
            prompt_key = 'coder'
        
        messages = [
            {
                "role": "system",
                "content": self.prompts[prompt_key]
            },
            {
                "role": "user",
                "content": f"""TECHNICAL_ANALYSIS:
{analysis}

CONTEXT_PACK:
{context_pack_with_ids}

Original User Request: {user_request}

Generate the complete, production-ready multi-file application following all grounding rules and artifact requirements. Include the REFERENCE_MAP_CONFIRMATION section before the code."""
            }
        ]
        
        model_name = self.models.get(model_type, self.models['claude'])
        response = self._call_llm(model_name, messages, temperature=0.1)
        
        # Create descriptive generator names
        generator_names = {
            'claude': 'Claude Sonnet 4 (Production-Ready)',
            'gpt': 'GPT-4o (Enterprise-Grade)', 
            'gemini': 'Gemini 2.5 Pro (Analytical)'
        }
        
        # Extract pure Python code from response
        python_code = self._extract_python_code(response)
        
        return CodeSolution(
            generator=generator_names.get(model_type, f"{model_type.title()} (Consolidated)"),
            code=python_code,
            explanation=f"Generated with consolidated grounding enforcement: {assembled_context.summary}"
        )
    
    def _generate_code_claude(self, user_request: str, analysis: str) -> CodeSolution:
        """Generate code using Claude with consolidated prompts"""
        return self._generate_code_consolidated(user_request, analysis, 'claude')
    
    def _generate_code_gpt(self, user_request: str, analysis: str) -> CodeSolution:
        """Generate code using GPT-4o with consolidated prompts"""
        return self._generate_code_consolidated(user_request, analysis, 'gpt')
    
    def _generate_code_gemini(self, user_request: str, analysis: str) -> CodeSolution:
        """Generate code using Gemini 2.5 Pro with analytical focus"""
        return self._generate_code_consolidated(user_request, analysis, 'gemini')
    
    def _generate_code_parallel(self, user_request: str, analysis: str) -> List[CodeSolution]:
        """Generate code with all LLMs in parallel"""
        
        # Show which specific models we're using - concise single-line format
        self._print_progress("Generating code with Claude, GPT-4o, and Gemini in parallel...", "GENERATE", indent=1)
        
        def generate_with_timing(model_type: str) -> CodeSolution:
            """Generate code and measure timing for a specific model"""
            start_time = time.time()
            display_name = self._get_model_display_name(model_type)
            technical_name = self._get_model_technical_name(model_type)
            
            try:
                solution = self._generate_code_consolidated(user_request, analysis, model_type)
                generation_time = time.time() - start_time
                solution.generation_time = generation_time
                
                return solution
                
            except Exception as e:
                generation_time = time.time() - start_time
                
                # Check if we need to use backup model
                backup_reason = self._determine_backup_reason(str(e))
                if backup_reason:
                    self._print_progress(
                        f"⚠️ {display_name} failed ({backup_reason}), trying backup...", 
                        "ERROR", 
                        indent=2
                    )
                    # For now, just return error - backup logic can be added later
                
                self._print_progress(
                    f"❌ {display_name} failed after {self._format_time(generation_time)}: {backup_reason or str(e)}", 
                    "ERROR", 
                    indent=2
                )
                
                return CodeSolution(
                    generator=f"{display_name} (Error)",
                    code=f"# Error generating code with {display_name}: {str(e)}",
                    explanation=f"Generation failed: {str(e)}",
                    score=0.0,
                    generation_time=generation_time
                )
        
        # Execute all models in parallel
        models_to_run = ['claude', 'gpt', 'gemini']
        total_start = time.time()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(generate_with_timing, model): model 
                for model in models_to_run
            }
            
            solutions = []
            completed_count = 0
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                completed_count += 1
                
                try:
                    solution = future.result()
                    solutions.append(solution)
                    
                    # Single line completion message with time (remove emoji since _print_progress adds one)
                    self._print_progress(
                        f"[{completed_count}/3] {solution.generator} completed in {self._format_time(solution.generation_time)}", 
                        "SUCCESS", 
                        indent=1
                    )
                    
                except Exception as e:
                    self._print_progress(f"❌ [{completed_count}/3] {model} failed: {e}", "ERROR", indent=1)
        
        total_parallel_time = time.time() - total_start
        
        # Calculate efficiency metrics
        individual_times = [sol.generation_time for sol in solutions if sol.generation_time > 0]
        sequential_time_estimate = sum(individual_times) if individual_times else total_parallel_time
        parallel_efficiency = sequential_time_estimate / total_parallel_time if total_parallel_time > 0 else 1.0
        
        # Combined timing output on single line
        self._print_progress(
            f"Parallel generation completed in {self._format_time(total_parallel_time)} - Efficiency: {parallel_efficiency:.1f}x speedup vs sequential", 
            "TIMING"
        )
        
        # Sort solutions by generation order (claude, gpt, gemini)
        model_order = {'claude': 0, 'gpt': 1, 'gemini': 2}
        solutions.sort(key=lambda sol: model_order.get(
            sol.generator.split()[0].lower(), 99
        ))
        
        return solutions
    
    def _determine_backup_reason(self, error_message: str) -> str:
        """Determine the reason for needing a backup model"""
        error_lower = error_message.lower()
        
        if "rate limit" in error_lower or "quota" in error_lower:
            return "rate limit exceeded"
        elif "timeout" in error_lower:
            return "request timeout"
        elif "authentication" in error_lower or "unauthorized" in error_lower:
            return "authentication failed"
        elif "model not found" in error_lower or "not available" in error_lower:
            return "model unavailable"
        elif "network" in error_lower or "connection" in error_lower:
            return "network error"
        else:
            return None  # Generic error, no specific backup reason
    
    def _score_solutions(self, solutions: List[CodeSolution], user_request: str, analysis: str = "") -> List[CodeSolution]:
        """Score and rank code solutions using consolidated evaluation with strict grounding checks"""
        
        def score_single_solution(solution: CodeSolution) -> CodeSolution:
            """Score a single solution and return it with score"""
            start_time = time.time()
            
            messages = [
                {
                    "role": "system",
                    "content": self.prompts['scorer']
                },
                {
                    "role": "user",
                    "content": f"""GENERATED_CODE:
{solution.code}

TECHNICAL_ANALYSIS:
{analysis}

CONTEXT_PACK:
[Available through the generated code's reference map]

Original Request: {user_request}
Generator: {solution.generator}

Perform rigorous evaluation following the YAML output format. Check for grounding violations and missing artifacts that would trigger auto-fail conditions."""
                }
            ]
            
            response = self._call_llm(self.models['scorer'], messages, temperature=0.0)
            
            # Parse YAML-structured response for score and verdict
            score, verdict = self._parse_scorer_response(response)
            
            solution.score = score
            solution.explanation += f"\\n\\n=== CONSOLIDATED EVALUATION ===\\nScore: {solution.score}/10\\nVerdict: {verdict}\\n{response}"
            
            eval_time = time.time() - start_time
            
            # Show score immediately (remove emoji since _print_progress adds one)
            self._print_progress(
                f"{solution.generator}: {solution.score}/10 ({verdict}) - evaluated in {self._format_time(eval_time)}", 
                "SCORE", 
                indent=1
            )
            
            return solution
        
        # Score solutions in parallel and show results as they complete
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_solution = {
                executor.submit(score_single_solution, solution): solution 
                for solution in solutions
            }
            
            scored_solutions = []
            for future in as_completed(future_to_solution):
                try:
                    scored_solution = future.result()
                    scored_solutions.append(scored_solution)
                except Exception as e:
                    original_solution = future_to_solution[future]
                    self._print_progress(f"❌ Scoring failed for {original_solution.generator}: {e}", "ERROR", indent=2)
                    # Keep original solution with 0 score
                    original_solution.score = 0.0
                    scored_solutions.append(original_solution)
        
        # Sort by score (highest first), but prioritize PASS verdicts
        scored_solutions.sort(key=lambda x: (x.score, 'PASS' in x.explanation), reverse=True)
        
        return scored_solutions
    
    def _parse_scorer_response(self, response: str) -> tuple[float, str]:
        """Parse the YAML-structured scorer response to extract score and verdict"""
        import re
        import yaml
        
        # Try to parse as YAML first
        try:
            # Extract YAML block if wrapped in code fence
            yaml_match = re.search(r'```yaml\s*\n(.*?)```', response, re.DOTALL)
            if yaml_match:
                yaml_text = yaml_match.group(1)
            else:
                yaml_text = response
                
            parsed = yaml.safe_load(yaml_text)
            
            if isinstance(parsed, dict):
                score = float(parsed.get('SCORE', 0.0))
                verdict = parsed.get('VERDICT', 'FAIL')
                
                # Handle auto-fail conditions
                if verdict == 'FAIL':
                    score = 0.0
                    
                return min(max(score, 0.0), 10.0), verdict
                
        except (yaml.YAMLError, ValueError, AttributeError):
            pass
        
        # Fallback to regex parsing
        verdict_match = re.search(r'VERDICT:\s*["\']?(PASS|FAIL)["\']?', response, re.IGNORECASE)
        verdict = verdict_match.group(1) if verdict_match else "FAIL"
        
        score_match = re.search(r'SCORE:\s*([0-9]+\.?[0-9]*)', response, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            # Auto-fail conditions override the score
            if verdict == "FAIL":
                score = 0.0
        else:
            # Default score based on verdict
            score = 0.0 if verdict == "FAIL" else 5.0
        
        # Check for specific failure indicators
        failure_indicators = [
            "hallucination", "not grounded", "missing artifact", 
            "auto-fail", "violation", "not found in context"
        ]
        
        if any(indicator in response.lower() for indicator in failure_indicators):
            score = 0.0
            verdict = "FAIL"
            
        return min(max(score, 0.0), 10.0), verdict
    
    def generate_code(self, user_request: str, enable_synthesis: bool = True) -> Dict:
        """Generate DataRobot API code using parallel LLMs with enhanced context and hybrid synthesis"""
        
        # Initialize metrics tracking
        metrics = GenerationMetrics()
        total_start_time = time.time()
        
        try:
            self._print_progress(f"Generating DataRobot code for: {user_request[:100]}{'...' if len(user_request) > 100 else ''}", "INIT")
            
            # Phase 1: Context Assembly & Analysis
            self._print_progress("Phase 1: Context Assembly & Requirements Analysis", "ANALYSIS")
            
            analysis_start = time.time()
            
            # Get enhanced context preview for logging
            self._print_progress("Searching for relevant DataRobot documentation and examples...", "CONTEXT", indent=1)
            context_preview = self.context_assembler.assemble_context(
                query=user_request,
                model_name='preview',
                max_examples=2,
                complexity_preference='medium'
            )
            
            self._print_progress(f"Found: {len(context_preview.code_examples)} code examples, {len(context_preview.api_references)} API references from {', '.join(context_preview.source_types)}", "CONTEXT", indent=2)
            
            # Analyze requirements
            self._print_progress(f"Analyzing requirements...", "ANALYSIS", indent=1)
            analysis, analysis_time = self._measure_time(self._analyze_requirements, user_request)
            metrics.analysis_time = analysis_time
            
            self._print_progress(f"Requirements analyzed in {self._format_time(analysis_time)}", "TIMING", indent=2)
            
            # Phase 2: Parallel Code Generation
            self._print_progress("Phase 2: Parallel Multi-LLM Code Generation", "GENERATE")
            generation_start = time.time()
            
            # Generate solutions in parallel
            solutions = self._generate_code_parallel(user_request, analysis)
            generation_time = time.time() - generation_start
            metrics.generation_time = generation_time
            
            # Phase 3: Solution Scoring & Evaluation  
            self._print_progress("Phase 3: Solution Scoring & Evaluation", "SCORE")
            scoring_start = time.time()
            
            scored_solutions, scoring_time = self._measure_time(
                self._score_solutions, solutions, user_request, analysis
            )
            metrics.scoring_time = scoring_time
            
            # Scoring results are already displayed in real-time within _score_solutions
            # Remove the "All evaluations completed" line to reduce verbosity - timing is already shown per evaluation
            
            # Phase 4: Hybrid Code Synthesis (if enabled)
            final_solution = scored_solutions[0]  # Default to best scored solution
            synthesis_metrics = None
            
            if enable_synthesis:
                self._print_progress("Phase 4: Hybrid Code Synthesis", "SYNTHESIZE")
                synthesis_start = time.time()
                
                try:
                    synthesis_result, synthesis_metrics = self.synthesis_orchestrator.orchestrate_synthesis(
                        scored_solutions, user_request
                    )
                    
                    # Check if synthesis improved the solution (synthesis orchestrator already printed status)
                    if synthesis_result.final_score_estimate > final_solution.score:
                        # Create enhanced solution from synthesis result with hybrid details
                        contributing_models = list(synthesis_result.source_contributions.keys())
                        hybrid_description = f"Hybrid of {', '.join(contributing_models)} ({synthesis_result.strategy_used.value})"
                        
                        final_solution = CodeSolution(
                            generator=hybrid_description,
                            code=synthesis_result.synthesized_code,
                            explanation=synthesis_result.synthesis_reasoning,
                            score=synthesis_result.final_score_estimate,
                            generation_time=synthesis_metrics.total_synthesis_time
                        )
                    # If synthesis didn't improve, keep original solution (orchestrator already reported this)
                    
                    metrics.synthesis_time = time.time() - synthesis_start
                    
                except Exception as e:
                    self._print_progress(f"Synthesis failed, using best scored solution: {e}", "ERROR", indent=1)
                    metrics.synthesis_time = time.time() - synthesis_start
            else:
                self._print_progress("Synthesis disabled, using best scored solution", "INFO")
            
            # Phase 5: Results & Metrics
            metrics.total_time = time.time() - total_start_time
            
            # Calculate parallel efficiency
            individual_times = [sol.generation_time for sol in solutions if sol.generation_time > 0]
            if individual_times:
                sequential_estimate = sum(individual_times)
                actual_parallel_time = max(individual_times)  # Slowest determines parallel time
                metrics.parallel_efficiency = sequential_estimate / actual_parallel_time
            
            # Phase 5: Results Summary (streamlined to avoid duplication)
            self._print_progress(f"Final solution: {final_solution.generator} (Score: {final_solution.score}/10)", "SUCCESS")
            self._print_progress(f"Total time: {self._format_time(metrics.total_time)} - Parallel efficiency: {metrics.parallel_efficiency:.1f}x speedup", "TIMING", indent=1)
            
            if synthesis_metrics:
                self._print_progress(f"Synthesis time: {self._format_time(synthesis_metrics.total_synthesis_time)} - Quality improvement: +{synthesis_metrics.quality_improvement:.1f}", "TIMING", indent=1)
            
            # Build comprehensive result
            return {
                'status': 'success',
                'best_solution': {
                    'generator': final_solution.generator,
                    'code': final_solution.code,
                    'score': final_solution.score,
                    'explanation': final_solution.explanation,
                    'generation_time': final_solution.generation_time
                },
                'all_solutions': [
                    {
                        'generator': sol.generator,
                        'score': sol.score,
                        'generation_time': sol.generation_time,
                        'code_length': len(sol.code),
                        'code_preview': sol.code[:200] + "..." if len(sol.code) > 200 else sol.code
                    }
                    for sol in scored_solutions
                ],
                'metrics': {
                    'total_time': metrics.total_time,
                    'analysis_time': metrics.analysis_time,
                    'generation_time': metrics.generation_time,
                    'scoring_time': metrics.scoring_time,
                    'synthesis_time': metrics.synthesis_time,
                    'parallel_efficiency': metrics.parallel_efficiency,
                    'models_executed': len([s for s in solutions if s.score > 0 or 'Error' not in s.generator]),
                    'synthesis_enabled': enable_synthesis
                },
                'synthesis_info': synthesis_metrics.__dict__ if synthesis_metrics else None,
                'models_used': list(self.models.values()),
                'analysis': analysis,
                'enhanced_context': {
                    'summary': context_preview.summary,
                    'content_types': context_preview.content_types,
                    'source_types': context_preview.source_types,
                    'query_type': context_preview.query_type.value if hasattr(context_preview, 'query_type') else 'unknown',
                    'total_sources': len(context_preview.source_citations) if hasattr(context_preview, 'source_citations') else 0
                }
            }
            
        except Exception as e:
            metrics.total_time = time.time() - total_start_time
            self._print_progress(f"Generation failed after {self._format_time(metrics.total_time)}: {str(e)}", "ERROR")
            
            return {
                'status': 'error',
                'error': str(e),
                'partial_metrics': {
                    'total_time': metrics.total_time,
                    'analysis_time': metrics.analysis_time,
                    'generation_time': metrics.generation_time,
                    'scoring_time': metrics.scoring_time
                },
                'models_used': list(self.models.values()),
                'enhanced_retrieval_enabled': hasattr(self, 'context_assembler')
            }

def test_multi_llm():
    """Test the multi-LLM system"""
    
    # Create retriever with sample data 
    import sys
    sys.path.append('/Users/jeremy.pernicek/Desktop/DR_API_AGENT')
    from agent_crewai.src.rag.hybrid_retriever import HybridRetriever, create_sample_data
    
    retriever = HybridRetriever()
    sample_docs = create_sample_data()
    retriever.add_documents(sample_docs)
    retriever.build_indexes()
    
    # Create multi-LLM system
    multi_llm = SimpleMultiLLM(retriever=retriever)
    
    # Test request
    test_request = "Create a DataRobot project from a CSV file, set up time series forecasting with a 7-day forecast window, and start autopilot"
    
    print("🚀 Testing Multi-LLM DataRobot Code Generation...")
    result = multi_llm.generate_code(test_request)
    
    if result['status'] == 'success':
        print(f"\\n🎉 Best Solution from {result['best_solution']['generator']} (Score: {result['best_solution']['score']}):")
        print("\\n" + "="*80)
        print(result['best_solution']['code'])
        print("="*80)
        
        print(f"\\n📊 All Solutions Compared:")
        for sol in result['all_solutions']:
            print(f"  - {sol['generator']}: {sol['score']}/10")
            
    else:
        print(f"❌ Error: {result['error']}")
    
    return result

if __name__ == "__main__":
    test_multi_llm()