"""
DataRobot Code Synthesizer - Intelligent hybrid synthesis of multi-LLM generated code.
Combines the best components from multiple solutions into optimized final code.
"""

import re
import ast
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Import shared models to avoid circular imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from shared_models import CodeSolution, SynthesisStrategy


@dataclass
class SolutionAnalysis:
    """Analysis of a code solution's components and quality"""
    solution: CodeSolution
    code_structure: Dict[str, str]    # imports, setup, workflow, results, etc.
    strengths: List[str]              # What this solution does well
    weaknesses: List[str]             # What could be improved
    extractable_components: Dict[str, str]  # Reusable parts
    quality_score: float              # 0-10 overall quality assessment
    

@dataclass
class SynthesisResult:
    """Result of code synthesis process"""
    synthesized_code: str
    strategy_used: SynthesisStrategy
    source_contributions: Dict[str, List[str]]  # Which parts came from which LLMs
    quality_improvements: List[str]
    synthesis_reasoning: str
    final_score_estimate: float


class DataRobotCodeSynthesizer:
    """Intelligent code synthesis system for DataRobot API solutions"""
    
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 8.5,      # Use best only
            'good': 6.5,           # Hybrid synthesis beneficial  
            'poor': 4.0,           # Recovery mode
        }
        
        # Patterns for extracting code components
        self.section_patterns = {
            'imports': r'^(import\s+.*?|from\s+.*?import.*?)$',
            'setup': r'# 📦.*?SECTION.*?(?=# [🚀📊]|$)',
            'workflow': r'# 🚀.*?SECTION.*?(?=# 📊|$)', 
            'results': r'# 📊.*?SECTION.*?(?=# |$)',
            'error_handling': r'try:\s*.*?except.*?:.*?(?=\n\n|\n#|$)',
            'functions': r'def\s+\w+\(.*?\):.*?(?=\ndef|\n#|\nif|\Z)',
            'classes': r'class\s+\w+.*?:.*?(?=\nclass|\ndef|\n#|\nif|\Z)',
            'comments': r'#.*$'
        }
    
    def decide_strategy(self, solutions: List[CodeSolution]) -> Tuple[SynthesisStrategy, str]:
        """Decide which synthesis strategy to use based on solution quality"""
        
        if not solutions:
            return SynthesisStrategy.RECOVERY_MODE, "No solutions provided"
        
        # Filter out error solutions
        valid_solutions = [sol for sol in solutions if sol.score > 0 and 'Error' not in sol.generator]
        
        if not valid_solutions:
            return SynthesisStrategy.RECOVERY_MODE, "All solutions failed - attempting recovery"
        
        # Get scores
        scores = [sol.score for sol in valid_solutions]
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        # Decision logic
        if best_score >= self.quality_thresholds['excellent']:
            if score_variance < 1.0:  # All solutions are similarly excellent
                strategy = SynthesisStrategy.HYBRID_SYNTHESIS
                reason = f"Multiple excellent solutions (best: {best_score:.1f}, avg: {avg_score:.1f}) - hybrid beneficial"
            else:
                strategy = SynthesisStrategy.BEST_ONLY  
                reason = f"One clearly superior solution (score: {best_score:.1f}) - use as-is"
                
        elif best_score >= self.quality_thresholds['good']:
            if len(valid_solutions) > 1 and score_variance > 0.5:
                strategy = SynthesisStrategy.ENHANCED_BEST
                reason = f"Good best solution (score: {best_score:.1f}) can be enhanced with components from others"
            else:
                strategy = SynthesisStrategy.HYBRID_SYNTHESIS
                reason = f"Multiple good solutions (avg: {avg_score:.1f}) - combine strengths"
                
        else:
            strategy = SynthesisStrategy.RECOVERY_MODE
            reason = f"All solutions suboptimal (best: {best_score:.1f}) - attempting recovery synthesis"
        
        return strategy, reason
    
    def analyze_solutions(self, solutions: List[CodeSolution]) -> List[SolutionAnalysis]:
        """Analyze each solution to understand its components and quality"""
        analyses = []
        
        for solution in solutions:
            analysis = SolutionAnalysis(
                solution=solution,
                code_structure=self._extract_code_structure(solution.code),
                strengths=self._identify_strengths(solution),
                weaknesses=self._identify_weaknesses(solution),
                extractable_components=self._extract_components(solution.code),
                quality_score=solution.score
            )
            analyses.append(analysis)
        
        return analyses
    
    def synthesize_components(self, analyses: List[SolutionAnalysis], strategy: SynthesisStrategy) -> SynthesisResult:
        """Synthesize components based on chosen strategy"""
        
        if strategy == SynthesisStrategy.BEST_ONLY:
            return self._synthesize_best_only(analyses)
        elif strategy == SynthesisStrategy.HYBRID_SYNTHESIS:
            return self._synthesize_hybrid(analyses)
        elif strategy == SynthesisStrategy.ENHANCED_BEST:
            return self._synthesize_enhanced_best(analyses)
        else:  # RECOVERY_MODE
            return self._synthesize_recovery(analyses)
    
    def enhance_best(self, synthesis_result: SynthesisResult, analyses: List[SolutionAnalysis]) -> SynthesisResult:
        """Apply final enhancements to the synthesized code"""
        
        enhanced_code = synthesis_result.synthesized_code
        improvements = list(synthesis_result.quality_improvements)
        
        # Apply enhancement patterns
        enhanced_code = self._optimize_imports(enhanced_code)
        enhanced_code = self._improve_error_handling(enhanced_code, analyses)
        enhanced_code = self._enhance_comments(enhanced_code, analyses)
        enhanced_code = self._optimize_structure(enhanced_code)
        
        # Track improvements
        improvements.extend([
            "Optimized import statements",
            "Enhanced error handling patterns", 
            "Improved code documentation",
            "Optimized code structure"
        ])
        
        return SynthesisResult(
            synthesized_code=enhanced_code,
            strategy_used=synthesis_result.strategy_used,
            source_contributions=synthesis_result.source_contributions,
            quality_improvements=improvements,
            synthesis_reasoning=synthesis_result.synthesis_reasoning + " Enhanced with optimization patterns.",
            final_score_estimate=min(synthesis_result.final_score_estimate + 0.5, 10.0)  # Modest improvement
        )
    
    def _extract_code_structure(self, code: str) -> Dict[str, str]:
        """Extract major structural components from code"""
        structure = {}
        
        # Extract sections using patterns
        for section_name, pattern in self.section_patterns.items():
            matches = re.findall(pattern, code, re.MULTILINE | re.DOTALL)
            if matches:
                if section_name in ['imports', 'comments']:
                    structure[section_name] = '\n'.join(matches)
                else:
                    structure[section_name] = matches[0] if matches else ""
            else:
                structure[section_name] = ""
        
        return structure
    
    def _identify_strengths(self, solution: CodeSolution) -> List[str]:
        """Identify strengths of a solution"""
        strengths = []
        code = solution.code
        
        # Check for good practices
        if 'import datarobot as dr' in code:
            strengths.append("Proper DataRobot SDK import")
        
        if '# 📦' in code and '# 🚀' in code:
            strengths.append("Clear section organization")
        
        if 'try:' in code and 'except' in code:
            strengths.append("Error handling implemented")
        
        if 'print(' in code:
            strengths.append("Progress indicators included")
        
        if solution.score >= 8.0:
            strengths.append("High overall quality score")
        
        # Check for DataRobot best practices
        if 'dr.Client(' in code:
            strengths.append("Proper client initialization")
        
        if 'project.wait_for_autopilot()' in code:
            strengths.append("Proper autopilot handling")
        
        if len(code.split('\n')) < 100:
            strengths.append("Concise implementation")
        elif len(code.split('\n')) > 100:
            strengths.append("Comprehensive implementation")
        
        return strengths
    
    def _identify_weaknesses(self, solution: CodeSolution) -> List[str]:
        """Identify weaknesses of a solution"""
        weaknesses = []
        code = solution.code
        
        # Check for common issues
        if solution.score < 6.0:
            weaknesses.append("Low quality score")
        
        if 'Error' in solution.generator:
            weaknesses.append("Generation failed")
        
        if code.count("'''") > 4:
            weaknesses.append("Too much verbose text")
        
        if '@dataclass' in code and 'AnalysisStep' in code:
            weaknesses.append("Contains verbose analytical frameworks")
        
        if code.count('#') > len(code.split('\n')) * 0.5:
            weaknesses.append("Excessive comments relative to code")
        
        if 'try:' not in code:
            weaknesses.append("Missing error handling")
        
        if 'print(' not in code:
            weaknesses.append("No progress indicators")
        
        return weaknesses
    
    def _extract_components(self, code: str) -> Dict[str, str]:
        """Extract reusable components from code"""
        components = {}
        
        # Extract imports
        import_lines = [line for line in code.split('\n') if line.strip().startswith(('import ', 'from '))]
        components['imports'] = '\n'.join(import_lines)
        
        # Extract functions
        func_matches = re.findall(self.section_patterns['functions'], code, re.DOTALL)
        components['functions'] = '\n\n'.join(func_matches) if func_matches else ""
        
        # Extract error handling patterns
        try_blocks = re.findall(self.section_patterns['error_handling'], code, re.DOTALL)
        components['error_handling'] = '\n\n'.join(try_blocks) if try_blocks else ""
        
        # Extract configuration sections
        config_pattern = r'# Configuration.*?\n(.*?)(?=\n\n|try:|def |class |#|$)'
        config_matches = re.findall(config_pattern, code, re.DOTALL)
        components['configuration'] = '\n'.join(config_matches) if config_matches else ""
        
        return components
    
    def _synthesize_best_only(self, analyses: List[SolutionAnalysis]) -> SynthesisResult:
        """Use the best solution as-is"""
        best_analysis = max(analyses, key=lambda a: a.quality_score)
        
        return SynthesisResult(
            synthesized_code=best_analysis.solution.code,
            strategy_used=SynthesisStrategy.BEST_ONLY,
            source_contributions={best_analysis.solution.generator: ["Complete solution"]},
            quality_improvements=["Selected highest quality solution"],
            synthesis_reasoning=f"Best solution ({best_analysis.solution.generator}, score: {best_analysis.quality_score:.1f}) used as-is due to high quality.",
            final_score_estimate=best_analysis.quality_score
        )
    
    def _synthesize_hybrid(self, analyses: List[SolutionAnalysis]) -> SynthesisResult:
        """Combine best components from multiple solutions"""
        # Start with the highest-scoring valid solution as base
        valid_analyses = [a for a in analyses if a.quality_score > 0]
        if not valid_analyses:
            return self._synthesize_recovery(analyses)
        
        base_analysis = max(valid_analyses, key=lambda a: a.quality_score)
        synthesized_code = base_analysis.solution.code
        
        source_contributions = {base_analysis.solution.generator: ["Base structure"]}
        improvements = ["Used best solution as foundation"]
        
        # Enhance with components from other solutions
        for analysis in valid_analyses:
            if analysis == base_analysis:
                continue
                
            generator = analysis.solution.generator
            if generator not in source_contributions:
                source_contributions[generator] = []
            
            # Look for better error handling
            if ('error_handling' in analysis.extractable_components and 
                analysis.extractable_components['error_handling'] and
                'try:' not in synthesized_code):
                
                error_handling = analysis.extractable_components['error_handling']
                # Insert error handling into synthesized code
                synthesized_code = self._merge_error_handling(synthesized_code, error_handling)
                source_contributions[generator].append("Error handling patterns")
                improvements.append("Enhanced error handling")
            
            # Look for better imports
            if ('imports' in analysis.extractable_components and 
                len(analysis.extractable_components['imports'].split('\n')) > 
                len(base_analysis.extractable_components.get('imports', '').split('\n'))):
                
                synthesized_code = self._merge_imports(synthesized_code, analysis.extractable_components['imports'])
                source_contributions[generator].append("Import statements")
                improvements.append("Optimized imports")
            
            # Look for additional configuration
            if ('configuration' in analysis.extractable_components and 
                analysis.extractable_components['configuration'] and
                'configuration' not in base_analysis.extractable_components):
                
                config = analysis.extractable_components['configuration']
                synthesized_code = self._merge_configuration(synthesized_code, config)
                source_contributions[generator].append("Configuration settings")
                improvements.append("Added configuration options")
        
        # Estimate final score
        avg_score = sum(a.quality_score for a in valid_analyses) / len(valid_analyses)
        final_score = min(base_analysis.quality_score + 0.5, avg_score + 1.0, 10.0)
        
        return SynthesisResult(
            synthesized_code=synthesized_code,
            strategy_used=SynthesisStrategy.HYBRID_SYNTHESIS,
            source_contributions=source_contributions,
            quality_improvements=improvements,
            synthesis_reasoning=f"Combined best components from {len(valid_analyses)} solutions. Base: {base_analysis.solution.generator}",
            final_score_estimate=final_score
        )
    
    def _synthesize_enhanced_best(self, analyses: List[SolutionAnalysis]) -> SynthesisResult:
        """Enhance the best solution with select components from others"""
        valid_analyses = [a for a in analyses if a.quality_score > 0]
        if not valid_analyses:
            return self._synthesize_recovery(analyses)
        
        best_analysis = max(valid_analyses, key=lambda a: a.quality_score)
        synthesized_code = best_analysis.solution.code
        
        source_contributions = {best_analysis.solution.generator: ["Primary solution"]}
        improvements = ["Enhanced best solution with select improvements"]
        
        # Only add components that clearly improve the best solution
        for analysis in valid_analyses:
            if analysis == best_analysis:
                continue
            
            generator = analysis.solution.generator
            
            # Add missing error handling if best solution lacks it
            if ('try:' not in synthesized_code and 
                'try:' in analysis.solution.code and
                'except' in analysis.solution.code):
                
                error_pattern = re.search(r'try:\s*.*?except.*?:', analysis.solution.code, re.DOTALL)
                if error_pattern:
                    synthesized_code = self._enhance_with_error_handling(synthesized_code, error_pattern.group())
                    source_contributions[generator] = source_contributions.get(generator, []) + ["Error handling"]
                    improvements.append("Added missing error handling")
            
            # Add progress indicators if missing
            if ('print(' not in synthesized_code and 'print(' in analysis.solution.code):
                progress_lines = [line for line in analysis.solution.code.split('\n') if 'print(' in line]
                if progress_lines:
                    synthesized_code = self._enhance_with_progress_indicators(synthesized_code, progress_lines[:2])
                    source_contributions[generator] = source_contributions.get(generator, []) + ["Progress indicators"]
                    improvements.append("Added progress indicators")
        
        final_score = min(best_analysis.quality_score + 0.3, 10.0)  # Modest enhancement
        
        return SynthesisResult(
            synthesized_code=synthesized_code,
            strategy_used=SynthesisStrategy.ENHANCED_BEST,
            source_contributions=source_contributions,
            quality_improvements=improvements,
            synthesis_reasoning=f"Enhanced {best_analysis.solution.generator} (score: {best_analysis.quality_score:.1f}) with components from other solutions",
            final_score_estimate=final_score
        )
    
    def _synthesize_recovery(self, analyses: List[SolutionAnalysis]) -> SynthesisResult:
        """Attempt to create usable code when all solutions are poor"""
        # Find the least bad solution
        best_attempt = max(analyses, key=lambda a: max(a.quality_score, 0))
        
        synthesized_code = best_attempt.solution.code
        
        # If code is completely broken, create minimal template
        if 'Error' in best_attempt.solution.generator or best_attempt.quality_score == 0:
            synthesized_code = self._create_minimal_template()
        
        return SynthesisResult(
            synthesized_code=synthesized_code,
            strategy_used=SynthesisStrategy.RECOVERY_MODE,
            source_contributions={best_attempt.solution.generator: ["Recovery base"]},
            quality_improvements=["Created minimal working solution"],
            synthesis_reasoning="All solutions were poor quality - created recovery solution",
            final_score_estimate=3.0  # Conservative estimate for recovery
        )
    
    def _create_minimal_template(self) -> str:
        """Create minimal DataRobot template as fallback"""
        return '''# 📦 SECTION 1: Quick Setup (Copy First)
import datarobot as dr
import pandas as pd

# Connect to DataRobot
dr.Client(
    token='YOUR_API_TOKEN',
    endpoint='https://app.datarobot.com/api/v2'
)
print("✅ Connected to DataRobot")

# 🚀 SECTION 2: Main Workflow (Core Functionality)
# Replace with your specific use case
CSV_FILE_PATH = 'your_data.csv'
PROJECT_NAME = 'My Project'

try:
    # Create project from data
    project = dr.Project.create(
        sourcedata=CSV_FILE_PATH,
        project_name=PROJECT_NAME
    )
    print(f"✅ Project created: {project.id}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# 📊 SECTION 3: Results & Next Steps (Optional) 
# Add your specific workflow steps here
'''
    
    # Helper methods for merging components
    def _merge_error_handling(self, base_code: str, error_handling: str) -> str:
        """Merge error handling into base code"""
        # Simple approach - add error handling around main workflow
        lines = base_code.split('\n')
        
        # Find main workflow section
        workflow_start = -1
        for i, line in enumerate(lines):
            if '# 🚀' in line:
                workflow_start = i + 1
                break
        
        if workflow_start > 0:
            # Wrap workflow in try-except
            lines.insert(workflow_start, 'try:')
            # Find end of workflow and add except
            for i in range(workflow_start + 1, len(lines)):
                if lines[i].startswith('# ') and ('📊' in lines[i] or '📝' in lines[i]):
                    lines.insert(i, 'except Exception as e:')
                    lines.insert(i + 1, '    print(f"❌ Error: {e}")')
                    break
        
        return '\n'.join(lines)
    
    def _merge_imports(self, base_code: str, new_imports: str) -> str:
        """Merge additional imports into base code"""
        base_imports = set()
        new_import_lines = []
        
        # Extract existing imports
        for line in base_code.split('\n'):
            if line.strip().startswith(('import ', 'from ')):
                base_imports.add(line.strip())
        
        # Add new unique imports
        for line in new_imports.split('\n'):
            if line.strip() and line.strip() not in base_imports:
                new_import_lines.append(line)
        
        if new_import_lines:
            lines = base_code.split('\n')
            # Find where to insert (after existing imports)
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    insert_pos = i + 1
            
            # Insert new imports
            for new_import in reversed(new_import_lines):
                lines.insert(insert_pos, new_import)
            
            return '\n'.join(lines)
        
        return base_code
    
    def _merge_configuration(self, base_code: str, config: str) -> str:
        """Merge configuration into base code"""
        # Insert configuration after main workflow comment
        lines = base_code.split('\n')
        for i, line in enumerate(lines):
            if '# 🚀' in line:
                lines.insert(i + 1, '')
                lines.insert(i + 2, '# Configuration')
                lines.insert(i + 3, config)
                lines.insert(i + 4, '')
                break
        
        return '\n'.join(lines)
    
    def _optimize_imports(self, code: str) -> str:
        """Optimize import statements"""
        lines = code.split('\n')
        import_lines = []
        other_lines = []
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ')):
                import_lines.append(line)
            else:
                other_lines.append(line)
        
        # Sort and deduplicate imports
        import_lines = sorted(list(set(import_lines)))
        
        # Rebuild code
        if import_lines and other_lines:
            return '\n'.join(import_lines + [''] + other_lines)
        return code
    
    def _improve_error_handling(self, code: str, analyses: List[SolutionAnalysis]) -> str:
        """Improve error handling based on analysis of other solutions"""
        # If no try-except blocks, add basic error handling
        if 'try:' not in code:
            return self._add_basic_error_handling(code)
        return code
    
    def _add_basic_error_handling(self, code: str) -> str:
        """Add basic error handling to code"""
        lines = code.split('\n')
        
        # Find main workflow
        for i, line in enumerate(lines):
            if '# 🚀' in line:
                # Add try after the comment
                if i + 1 < len(lines):
                    lines.insert(i + 1, 'try:')
                    # Indent following lines until next section
                    j = i + 2
                    while j < len(lines) and not (lines[j].startswith('# ') and any(emoji in lines[j] for emoji in ['📊', '📝', '🔍'])):
                        if lines[j].strip():  # Don't indent empty lines
                            lines[j] = '    ' + lines[j]
                        j += 1
                    
                    # Add except clause
                    lines.insert(j, 'except Exception as e:')
                    lines.insert(j + 1, '    print(f"❌ Error: {e}")')
                break
        
        return '\n'.join(lines)
    
    def _enhance_comments(self, code: str, analyses: List[SolutionAnalysis]) -> str:
        """Enhance comments based on other solutions"""
        # Basic implementation - could be enhanced
        return code
    
    def _optimize_structure(self, code: str) -> str:
        """Optimize code structure"""
        # Remove excessive blank lines
        lines = code.split('\n')
        optimized_lines = []
        prev_empty = False
        
        for line in lines:
            if line.strip():
                optimized_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                optimized_lines.append(line)
                prev_empty = True
        
        return '\n'.join(optimized_lines)
    
    def _enhance_with_error_handling(self, code: str, error_pattern: str) -> str:
        """Add error handling pattern to code"""
        # Simple implementation
        return code.replace('project = dr.Project.create(', 'try:\n    project = dr.Project.create(') + '\nexcept Exception as e:\n    print(f"❌ Error: {e}")'
    
    def _enhance_with_progress_indicators(self, code: str, progress_lines: List[str]) -> str:
        """Add progress indicators to code"""
        lines = code.split('\n')
        
        # Add progress indicators after key operations
        for i, line in enumerate(lines):
            if 'dr.Client(' in line:
                lines.insert(i + 1, 'print("✅ Connected to DataRobot")')
            elif 'Project.create(' in line:
                lines.insert(i + 1, 'print("✅ Project created successfully")')
        
        return '\n'.join(lines)