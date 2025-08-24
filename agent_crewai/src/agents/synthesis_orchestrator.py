"""
Synthesis Orchestrator - Manages the 4-stage hybrid code synthesis pipeline.
Coordinates analysis, strategy selection, synthesis, and enhancement phases.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import required classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from shared_models import CodeSolution, SynthesisStrategy
from code_synthesizer import DataRobotCodeSynthesizer, SolutionAnalysis, SynthesisResult


class SynthesisStage(Enum):
    """Stages of the synthesis pipeline"""
    ANALYSIS = "analysis"           # Stage 1: Analyze individual solutions
    STRATEGY = "strategy"           # Stage 2: Decide synthesis strategy
    SYNTHESIS = "synthesis"         # Stage 3: Synthesize components
    ENHANCEMENT = "enhancement"     # Stage 4: Final enhancement and validation


@dataclass
class StageResult:
    """Result of a synthesis pipeline stage"""
    stage: SynthesisStage
    success: bool
    duration: float
    result_data: Dict
    stage_reasoning: str
    

@dataclass
class SynthesisMetrics:
    """Comprehensive metrics for synthesis pipeline"""
    total_synthesis_time: float = 0.0
    analysis_time: float = 0.0
    strategy_time: float = 0.0
    synthesis_time: float = 0.0
    enhancement_time: float = 0.0
    
    solutions_analyzed: int = 0
    strategy_confidence: float = 0.0
    components_synthesized: int = 0
    enhancements_applied: int = 0
    
    quality_improvement: float = 0.0  # Before vs after synthesis
    final_confidence: float = 0.0


class SynthesisOrchestrator:
    """Orchestrates the 4-stage hybrid code synthesis pipeline"""
    
    def __init__(self, synthesizer: Optional[DataRobotCodeSynthesizer] = None, verbose: bool = True):
        self.synthesizer = synthesizer or DataRobotCodeSynthesizer()
        self.verbose = verbose
        self.quiet_mode = False  # For concise output in production
    
    def set_quiet_mode(self, quiet: bool = True):
        """Enable/disable quiet mode for concise output"""
        self.quiet_mode = quiet
        
        # Pipeline configuration
        self.stage_config = {
            SynthesisStage.ANALYSIS: {
                'timeout': 30.0,
                'critical': True,
                'retry_on_failure': False
            },
            SynthesisStage.STRATEGY: {
                'timeout': 10.0,
                'critical': True,
                'retry_on_failure': True
            },
            SynthesisStage.SYNTHESIS: {
                'timeout': 45.0,
                'critical': True,
                'retry_on_failure': True
            },
            SynthesisStage.ENHANCEMENT: {
                'timeout': 20.0,
                'critical': False,  # Enhancement is optional
                'retry_on_failure': False
            }
        }
    
    def _print_stage_progress(self, message: str, stage: SynthesisStage, indent: int = 0):
        """Print progress for synthesis stages"""
        if not self.verbose or self.quiet_mode:
            return
            
        stage_emojis = {
            SynthesisStage.ANALYSIS: "🔬",
            SynthesisStage.STRATEGY: "🎯", 
            SynthesisStage.SYNTHESIS: "🧬",
            SynthesisStage.ENHANCEMENT: "✨"
        }
        
        emoji = stage_emojis.get(stage, "•")
        indent_str = "  " * indent
        print(f"{indent_str}{emoji} {message}")
    
    def _print_concise_progress(self, message: str):
        """Print concise progress message for quiet mode"""
        if self.verbose and not self.quiet_mode:
            return  # Don't double-print in verbose mode
        print(f"🧬 {message}")
    
    def orchestrate_synthesis(self, solutions: List[CodeSolution], user_request: str = "") -> Tuple[SynthesisResult, SynthesisMetrics]:
        """Execute the complete 4-stage synthesis pipeline"""
        
        metrics = SynthesisMetrics()
        total_start_time = time.time()
        
        # Concise output for production, verbose for development
        if self.quiet_mode:
            # Just show that synthesis is starting
            pass  # Will show final result at the end
        else:
            self._print_stage_progress("Starting 4-Stage Hybrid Code Synthesis", SynthesisStage.ANALYSIS)
            self._print_stage_progress(f"Input: {len(solutions)} solutions to synthesize", SynthesisStage.ANALYSIS, indent=1)
        
        try:
            # Stage 1: Solution Analysis & Quality Assessment  
            analysis_result = self._execute_stage_1_analysis(solutions, metrics)
            if not analysis_result.success:
                return self._create_fallback_result(solutions, metrics, "Analysis stage failed")
            
            analyses = analysis_result.result_data['analyses']
            
            # Stage 2: Strategy Decision
            strategy_result = self._execute_stage_2_strategy(analyses, metrics)
            if not strategy_result.success:
                return self._create_fallback_result(solutions, metrics, "Strategy selection failed")
            
            strategy = strategy_result.result_data['strategy']
            strategy_reasoning = strategy_result.result_data['reasoning']
            
            # Stage 3: Code Synthesis & Component Integration
            synthesis_result = self._execute_stage_3_synthesis(analyses, strategy, metrics)
            if not synthesis_result.success:
                return self._create_fallback_result(solutions, metrics, "Synthesis stage failed")
            
            initial_synthesis = synthesis_result.result_data['synthesis_result']
            
            # Stage 4: Final Enhancement & Validation (optional)
            enhancement_result = self._execute_stage_4_enhancement(initial_synthesis, analyses, metrics)
            
            # Use enhanced result if successful, otherwise use initial synthesis
            final_synthesis = (enhancement_result.result_data.get('enhanced_result', initial_synthesis) 
                             if enhancement_result.success else initial_synthesis)
            
            # Calculate final metrics
            metrics.total_synthesis_time = time.time() - total_start_time
            metrics.solutions_analyzed = len(solutions)
            
            # Calculate quality improvement
            original_best_score = max(sol.score for sol in solutions) if solutions else 0
            metrics.quality_improvement = final_synthesis.final_score_estimate - original_best_score
            metrics.final_confidence = self._calculate_confidence(final_synthesis, solutions)
            
            # Final output - concise for quiet mode, detailed for verbose
            if self.quiet_mode:
                # Show concise synthesis result
                if metrics.quality_improvement > 0:
                    # Show which models contributed and improvement
                    contributors = list(final_synthesis.source_contributions.keys())
                    main_contributors = [name.split()[0] for name in contributors[:2]]  # Just first names
                    if len(main_contributors) > 1:
                        self._print_concise_progress(f"Combining best components from {', '.join(main_contributors)}")
                    
                    original_best = max(sol.score for sol in solutions) if solutions else 0
                    self._print_concise_progress(f"Synthesis improved quality: {original_best:.1f} → {final_synthesis.final_score_estimate:.1f} ({strategy.value} strategy)")
                else:
                    # No improvement, show that original was kept
                    original_best = max(sol.score for sol in solutions) if solutions else 0
                    self._print_concise_progress(f"Original solution preferred: {original_best:.1f} vs {final_synthesis.final_score_estimate:.1f}")
            else:
                self._print_stage_progress("4-Stage Synthesis Pipeline Completed", SynthesisStage.ENHANCEMENT)
                self._print_stage_progress(f"Strategy: {strategy.value}", SynthesisStage.ENHANCEMENT, indent=1)
                self._print_stage_progress(f"Quality improvement: +{metrics.quality_improvement:.1f} points", SynthesisStage.ENHANCEMENT, indent=1)
                self._print_stage_progress(f"Total time: {metrics.total_synthesis_time:.2f}s", SynthesisStage.ENHANCEMENT, indent=1)
            
            return final_synthesis, metrics
            
        except Exception as e:
            metrics.total_synthesis_time = time.time() - total_start_time
            self._print_stage_progress(f"Pipeline failed: {e}", SynthesisStage.ANALYSIS)
            return self._create_fallback_result(solutions, metrics, f"Pipeline error: {e}")
    
    def _execute_stage_1_analysis(self, solutions: List[CodeSolution], metrics: SynthesisMetrics) -> StageResult:
        """Stage 1: Solution Analysis & Quality Assessment"""
        
        self._print_stage_progress("Stage 1: Solution Analysis & Quality Assessment", SynthesisStage.ANALYSIS)
        start_time = time.time()
        
        try:
            # Analyze each solution
            self._print_stage_progress(f"Analyzing {len(solutions)} individual solutions...", SynthesisStage.ANALYSIS, indent=1)
            
            analyses = self.synthesizer.analyze_solutions(solutions)
            
            # Log analysis results
            for analysis in analyses:
                self._print_stage_progress(
                    f"{analysis.solution.generator}: {len(analysis.strengths)} strengths, {len(analysis.weaknesses)} weaknesses", 
                    SynthesisStage.ANALYSIS, 
                    indent=2
                )
            
            duration = time.time() - start_time
            metrics.analysis_time = duration
            
            self._print_stage_progress(f"Analysis completed in {duration:.2f}s", SynthesisStage.ANALYSIS, indent=1)
            
            return StageResult(
                stage=SynthesisStage.ANALYSIS,
                success=True,
                duration=duration,
                result_data={'analyses': analyses},
                stage_reasoning=f"Successfully analyzed {len(analyses)} solutions"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.analysis_time = duration
            
            self._print_stage_progress(f"Analysis failed: {e}", SynthesisStage.ANALYSIS, indent=1)
            
            return StageResult(
                stage=SynthesisStage.ANALYSIS,
                success=False,
                duration=duration,
                result_data={'error': str(e)},
                stage_reasoning=f"Analysis failed: {e}"
            )
    
    def _execute_stage_2_strategy(self, analyses: List[SolutionAnalysis], metrics: SynthesisMetrics) -> StageResult:
        """Stage 2: Strategy Decision"""
        
        self._print_stage_progress("Stage 2: Strategy Decision", SynthesisStage.STRATEGY)
        start_time = time.time()
        
        try:
            # Extract solutions for strategy decision
            solutions = [analysis.solution for analysis in analyses]
            
            # Decide synthesis strategy
            strategy, reasoning = self.synthesizer.decide_strategy(solutions)
            
            # Calculate strategy confidence based on score distribution
            scores = [sol.score for sol in solutions if sol.score > 0]
            if scores:
                score_variance = sum((s - sum(scores)/len(scores)) ** 2 for s in scores) / len(scores)
                confidence = min(1.0, max(0.1, 1.0 - score_variance / 25.0))  # Normalize variance to confidence
            else:
                confidence = 0.1
            
            duration = time.time() - start_time
            metrics.strategy_time = duration
            metrics.strategy_confidence = confidence
            
            self._print_stage_progress(f"Strategy selected: {strategy.value}", SynthesisStage.STRATEGY, indent=1)
            self._print_stage_progress(f"Reasoning: {reasoning}", SynthesisStage.STRATEGY, indent=2)
            self._print_stage_progress(f"Confidence: {confidence:.2f}", SynthesisStage.STRATEGY, indent=2)
            self._print_stage_progress(f"Strategy decision completed in {duration:.3f}s", SynthesisStage.STRATEGY, indent=1)
            
            return StageResult(
                stage=SynthesisStage.STRATEGY,
                success=True,
                duration=duration,
                result_data={
                    'strategy': strategy,
                    'reasoning': reasoning,
                    'confidence': confidence
                },
                stage_reasoning=reasoning
            )
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.strategy_time = duration
            
            self._print_stage_progress(f"Strategy selection failed: {e}", SynthesisStage.STRATEGY, indent=1)
            
            return StageResult(
                stage=SynthesisStage.STRATEGY,
                success=False,
                duration=duration,
                result_data={'error': str(e)},
                stage_reasoning=f"Strategy selection failed: {e}"
            )
    
    def _execute_stage_3_synthesis(self, analyses: List[SolutionAnalysis], strategy: SynthesisStrategy, metrics: SynthesisMetrics) -> StageResult:
        """Stage 3: Code Synthesis & Component Integration"""
        
        self._print_stage_progress("Stage 3: Code Synthesis & Component Integration", SynthesisStage.SYNTHESIS)
        start_time = time.time()
        
        try:
            # Execute synthesis based on strategy
            self._print_stage_progress(f"Executing {strategy.value} synthesis...", SynthesisStage.SYNTHESIS, indent=1)
            
            synthesis_result = self.synthesizer.synthesize_components(analyses, strategy)
            
            # Count components that were synthesized
            components_count = sum(len(contributions) for contributions in synthesis_result.source_contributions.values())
            
            duration = time.time() - start_time
            metrics.synthesis_time = duration
            metrics.components_synthesized = components_count
            
            self._print_stage_progress(f"Synthesis completed with {components_count} components", SynthesisStage.SYNTHESIS, indent=1)
            self._print_stage_progress(f"Source contributions:", SynthesisStage.SYNTHESIS, indent=2)
            
            for generator, contributions in synthesis_result.source_contributions.items():
                self._print_stage_progress(f"  {generator}: {', '.join(contributions)}", SynthesisStage.SYNTHESIS, indent=2)
            
            self._print_stage_progress(f"Estimated quality: {synthesis_result.final_score_estimate:.1f}/10", SynthesisStage.SYNTHESIS, indent=2)
            self._print_stage_progress(f"Synthesis completed in {duration:.2f}s", SynthesisStage.SYNTHESIS, indent=1)
            
            return StageResult(
                stage=SynthesisStage.SYNTHESIS,
                success=True,
                duration=duration,
                result_data={'synthesis_result': synthesis_result},
                stage_reasoning=synthesis_result.synthesis_reasoning
            )
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.synthesis_time = duration
            
            self._print_stage_progress(f"Synthesis failed: {e}", SynthesisStage.SYNTHESIS, indent=1)
            
            return StageResult(
                stage=SynthesisStage.SYNTHESIS,
                success=False,
                duration=duration,
                result_data={'error': str(e)},
                stage_reasoning=f"Synthesis failed: {e}"
            )
    
    def _execute_stage_4_enhancement(self, synthesis_result: SynthesisResult, analyses: List[SolutionAnalysis], metrics: SynthesisMetrics) -> StageResult:
        """Stage 4: Final Enhancement & Validation (optional)"""
        
        self._print_stage_progress("Stage 4: Final Enhancement & Validation", SynthesisStage.ENHANCEMENT)
        start_time = time.time()
        
        try:
            # Apply enhancements to synthesized code
            self._print_stage_progress("Applying final enhancements...", SynthesisStage.ENHANCEMENT, indent=1)
            
            enhanced_result = self.synthesizer.enhance_best(synthesis_result, analyses)
            
            # Count enhancements applied
            new_improvements = len(enhanced_result.quality_improvements) - len(synthesis_result.quality_improvements)
            
            duration = time.time() - start_time
            metrics.enhancement_time = duration
            metrics.enhancements_applied = new_improvements
            
            # Calculate improvement
            quality_gain = enhanced_result.final_score_estimate - synthesis_result.final_score_estimate
            
            self._print_stage_progress(f"Applied {new_improvements} enhancements", SynthesisStage.ENHANCEMENT, indent=1)
            self._print_stage_progress(f"Quality improvement: +{quality_gain:.1f} points", SynthesisStage.ENHANCEMENT, indent=2)
            self._print_stage_progress(f"Final estimated score: {enhanced_result.final_score_estimate:.1f}/10", SynthesisStage.ENHANCEMENT, indent=2)
            self._print_stage_progress(f"Enhancement completed in {duration:.2f}s", SynthesisStage.ENHANCEMENT, indent=1)
            
            return StageResult(
                stage=SynthesisStage.ENHANCEMENT,
                success=True,
                duration=duration,
                result_data={'enhanced_result': enhanced_result, 'quality_gain': quality_gain},
                stage_reasoning=f"Applied {new_improvements} enhancements with +{quality_gain:.1f} quality improvement"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.enhancement_time = duration
            
            # Enhancement failure is not critical - log but continue
            self._print_stage_progress(f"Enhancement failed (non-critical): {e}", SynthesisStage.ENHANCEMENT, indent=1)
            self._print_stage_progress("Using original synthesis result", SynthesisStage.ENHANCEMENT, indent=2)
            
            return StageResult(
                stage=SynthesisStage.ENHANCEMENT,
                success=False,
                duration=duration,
                result_data={'error': str(e), 'fallback_result': synthesis_result},
                stage_reasoning=f"Enhancement failed (using original): {e}"
            )
    
    def _calculate_confidence(self, synthesis_result: SynthesisResult, original_solutions: List[CodeSolution]) -> float:
        """Calculate confidence in the synthesis result"""
        
        # Base confidence on strategy and quality
        strategy_confidence = {
            SynthesisStrategy.BEST_ONLY: 0.9,
            SynthesisStrategy.HYBRID_SYNTHESIS: 0.8,
            SynthesisStrategy.ENHANCED_BEST: 0.7,
            SynthesisStrategy.RECOVERY_MODE: 0.3
        }
        
        base_confidence = strategy_confidence.get(synthesis_result.strategy_used, 0.5)
        
        # Adjust based on estimated quality
        quality_factor = synthesis_result.final_score_estimate / 10.0
        
        # Adjust based on number of source contributions
        contribution_count = sum(len(contributions) for contributions in synthesis_result.source_contributions.values())
        diversity_factor = min(1.0, contribution_count / 5.0)  # More contributions = potentially better
        
        # Calculate final confidence
        final_confidence = (base_confidence * 0.5 + 
                           quality_factor * 0.3 + 
                           diversity_factor * 0.2)
        
        return min(1.0, max(0.1, final_confidence))
    
    def _create_fallback_result(self, solutions: List[CodeSolution], metrics: SynthesisMetrics, error_reason: str) -> Tuple[SynthesisResult, SynthesisMetrics]:
        """Create fallback result when pipeline fails"""
        
        # Use the best available solution as fallback
        if solutions:
            best_solution = max(solutions, key=lambda s: s.score)
            fallback_code = best_solution.code
            fallback_score = best_solution.score
            source_contrib = {best_solution.generator: ["Fallback solution"]}
        else:
            # Complete fallback
            fallback_code = "# Error: No valid solutions available"
            fallback_score = 0.0
            source_contrib = {"System": ["Emergency fallback"]}
        
        fallback_result = SynthesisResult(
            synthesized_code=fallback_code,
            strategy_used=SynthesisStrategy.RECOVERY_MODE,
            source_contributions=source_contrib,
            quality_improvements=[f"Fallback due to: {error_reason}"],
            synthesis_reasoning=f"Pipeline failed ({error_reason}), using fallback approach",
            final_score_estimate=fallback_score
        )
        
        metrics.final_confidence = 0.1  # Very low confidence for fallback
        
        return fallback_result, metrics
    
    def get_pipeline_summary(self, metrics: SynthesisMetrics) -> Dict:
        """Get a summary of the synthesis pipeline performance"""
        
        return {
            'total_time': metrics.total_synthesis_time,
            'stage_breakdown': {
                'analysis': metrics.analysis_time,
                'strategy': metrics.strategy_time, 
                'synthesis': metrics.synthesis_time,
                'enhancement': metrics.enhancement_time
            },
            'processing_stats': {
                'solutions_analyzed': metrics.solutions_analyzed,
                'components_synthesized': metrics.components_synthesized,
                'enhancements_applied': metrics.enhancements_applied
            },
            'quality_metrics': {
                'quality_improvement': metrics.quality_improvement,
                'strategy_confidence': metrics.strategy_confidence,
                'final_confidence': metrics.final_confidence
            },
            'performance': {
                'analysis_rate': metrics.solutions_analyzed / max(metrics.analysis_time, 0.001),
                'synthesis_efficiency': metrics.components_synthesized / max(metrics.synthesis_time, 0.001)
            }
        }