"""
Component Extractors - Utilities for extracting reusable components from DataRobot code solutions.
Provides specialized extractors for different types of code components.
"""

import re
import ast
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Import shared models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from shared_models import CodeSolution


class ComponentType(Enum):
    """Types of extractable components"""
    IMPORTS = "imports"
    ERROR_HANDLING = "error_handling"
    MONITORING = "monitoring"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    FUNCTIONS = "functions"
    CLASSES = "classes"
    WORKFLOW_SECTIONS = "workflow_sections"
    DATAROBOT_PATTERNS = "datarobot_patterns"


@dataclass 
class ExtractedComponent:
    """An extracted code component"""
    component_type: ComponentType
    content: str
    quality_score: float  # 0-10 assessment of component quality
    reusability_score: float  # 0-10 assessment of how reusable this component is
    source_generator: str
    extraction_confidence: float  # 0-1 confidence in extraction accuracy
    dependencies: List[str]  # Other components this depends on
    metadata: Dict  # Additional component-specific metadata


class ComponentExtractor:
    """Base class for component extractors"""
    
    def __init__(self):
        self.component_type = ComponentType.IMPORTS  # Override in subclasses
        
    def extract(self, code: str, source_generator: str = "unknown") -> List[ExtractedComponent]:
        """Extract components from code - override in subclasses"""
        raise NotImplementedError
    
    def _assess_quality(self, content: str, metadata: Dict) -> float:
        """Assess quality of extracted component"""
        # Base implementation - override for specific component types
        return 5.0
    
    def _assess_reusability(self, content: str, metadata: Dict) -> float:
        """Assess reusability of extracted component"""
        # Base implementation - override for specific component types
        return 5.0


class ImportExtractor(ComponentExtractor):
    """Extracts and analyzes import statements"""
    
    def __init__(self):
        super().__init__()
        self.component_type = ComponentType.IMPORTS
        
    def extract(self, code: str, source_generator: str = "unknown") -> List[ExtractedComponent]:
        """Extract import statements from code"""
        components = []
        
        # Find all import lines
        import_pattern = r'^((?:from\s+\S+\s+)?import\s+.+)$'
        import_lines = []
        
        for line in code.split('\n'):
            stripped = line.strip()
            if re.match(import_pattern, stripped):
                import_lines.append(stripped)
        
        if import_lines:
            # Group related imports
            grouped_imports = self._group_imports(import_lines)
            
            for group_name, imports in grouped_imports.items():
                content = '\n'.join(imports)
                metadata = {
                    'group': group_name,
                    'import_count': len(imports),
                    'has_datarobot': any('datarobot' in imp for imp in imports)
                }
                
                component = ExtractedComponent(
                    component_type=self.component_type,
                    content=content,
                    quality_score=self._assess_quality(content, metadata),
                    reusability_score=self._assess_reusability(content, metadata),
                    source_generator=source_generator,
                    extraction_confidence=0.95,  # High confidence for imports
                    dependencies=[],
                    metadata=metadata
                )
                components.append(component)
                
        return components
    
    def _group_imports(self, import_lines: List[str]) -> Dict[str, List[str]]:
        """Group related imports together"""
        groups = {
            'standard': [],
            'datarobot': [],
            'third_party': [],
            'pandas_numpy': []
        }
        
        for imp in import_lines:
            if 'datarobot' in imp.lower():
                groups['datarobot'].append(imp)
            elif any(lib in imp for lib in ['pandas', 'numpy']):
                groups['pandas_numpy'].append(imp)
            elif any(lib in imp for lib in ['os', 'sys', 'time', 'json', 'pathlib']):
                groups['standard'].append(imp)
            else:
                groups['third_party'].append(imp)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _assess_quality(self, content: str, metadata: Dict) -> float:
        """Assess quality of import statements"""
        score = 7.0  # Base score
        
        # Bonus for DataRobot imports
        if metadata.get('has_datarobot', False):
            score += 1.0
        
        # Bonus for organized imports
        if metadata.get('import_count', 0) > 1:
            score += 0.5
        
        # Penalty for too many imports in one group
        if metadata.get('import_count', 0) > 8:
            score -= 1.0
        
        return min(10.0, max(1.0, score))
    
    def _assess_reusability(self, content: str, metadata: Dict) -> float:
        """Assess reusability of import statements"""
        score = 8.0  # Imports are generally highly reusable
        
        # DataRobot imports are very reusable
        if metadata.get('has_datarobot', False):
            score += 1.5
        
        # Standard library imports are highly reusable  
        if metadata.get('group') == 'standard':
            score += 1.0
        
        return min(10.0, score)


class ErrorHandlingExtractor(ComponentExtractor):
    """Extracts error handling patterns"""
    
    def __init__(self):
        super().__init__()
        self.component_type = ComponentType.ERROR_HANDLING
    
    def extract(self, code: str, source_generator: str = "unknown") -> List[ExtractedComponent]:
        """Extract error handling patterns from code"""
        components = []
        
        # Pattern for try-except blocks
        try_except_pattern = r'try:\s*\n(.*?)\nexcept\s+(.*?):\s*\n(.*?)(?=\n\n|\ntry:|\ndef|\nclass|\n#|\Z)'
        
        matches = re.finditer(try_except_pattern, code, re.DOTALL | re.MULTILINE)
        
        for i, match in enumerate(matches):
            try_block = match.group(1).strip()
            except_clause = match.group(2).strip()
            except_block = match.group(3).strip()
            
            full_pattern = f"try:\n{try_block}\nexcept {except_clause}:\n{except_block}"
            
            metadata = {
                'exception_type': except_clause,
                'has_specific_exception': except_clause != 'Exception as e',
                'has_logging': 'print(' in except_block or 'log' in except_block.lower(),
                'try_block_length': len(try_block.split('\n')),
                'pattern_index': i
            }
            
            component = ExtractedComponent(
                component_type=self.component_type,
                content=full_pattern,
                quality_score=self._assess_quality(full_pattern, metadata),
                reusability_score=self._assess_reusability(full_pattern, metadata),
                source_generator=source_generator,
                extraction_confidence=0.85,
                dependencies=['imports'] if 'import' in try_block else [],
                metadata=metadata
            )
            components.append(component)
        
        return components
    
    def _assess_quality(self, content: str, metadata: Dict) -> float:
        """Assess quality of error handling"""
        score = 6.0
        
        # Bonus for specific exception types
        if metadata.get('has_specific_exception', False):
            score += 1.5
        
        # Bonus for logging/user feedback
        if metadata.get('has_logging', False):
            score += 1.0
        
        # Bonus for reasonable try block size
        try_length = metadata.get('try_block_length', 0)
        if 2 <= try_length <= 10:
            score += 1.0
        elif try_length > 15:
            score -= 1.0
        
        return min(10.0, max(1.0, score))
    
    def _assess_reusability(self, content: str, metadata: Dict) -> float:
        """Assess reusability of error handling"""
        score = 7.0
        
        # General exception handling is more reusable
        if not metadata.get('has_specific_exception', False):
            score += 1.0
        
        # Error handling with logging is more reusable
        if metadata.get('has_logging', False):
            score += 1.5
        
        return min(10.0, score)


class MonitoringExtractor(ComponentExtractor):
    """Extracts monitoring and progress tracking patterns"""
    
    def __init__(self):
        super().__init__()
        self.component_type = ComponentType.MONITORING
    
    def extract(self, code: str, source_generator: str = "unknown") -> List[ExtractedComponent]:
        """Extract monitoring patterns from code"""
        components = []
        
        # Find print statements for progress tracking
        print_pattern = r'print\(([^)]+)\)'
        print_matches = re.finditer(print_pattern, code)
        
        progress_prints = []
        for match in print_matches:
            print_content = match.group(0)
            if any(indicator in print_content.lower() for indicator in ['✅', '❌', '⏳', 'success', 'completed', 'error']):
                progress_prints.append(print_content)
        
        if progress_prints:
            content = '\n'.join(progress_prints)
            metadata = {
                'print_count': len(progress_prints),
                'has_emojis': any(emoji in content for emoji in ['✅', '❌', '⏳']),
                'progress_indicators': [p for p in progress_prints if 'progress' in p.lower() or '⏳' in p]
            }
            
            component = ExtractedComponent(
                component_type=self.component_type,
                content=content,
                quality_score=self._assess_quality(content, metadata),
                reusability_score=self._assess_reusability(content, metadata),
                source_generator=source_generator,
                extraction_confidence=0.8,
                dependencies=[],
                metadata=metadata
            )
            components.append(component)
        
        return components
    
    def _assess_quality(self, content: str, metadata: Dict) -> float:
        """Assess quality of monitoring patterns"""
        score = 6.0
        
        # Bonus for emojis (better UX)
        if metadata.get('has_emojis', False):
            score += 1.5
        
        # Bonus for progress indicators
        progress_count = len(metadata.get('progress_indicators', []))
        if progress_count > 0:
            score += min(2.0, progress_count * 0.5)
        
        # Penalty for too many print statements
        if metadata.get('print_count', 0) > 10:
            score -= 1.0
        
        return min(10.0, max(1.0, score))
    
    def _assess_reusability(self, content: str, metadata: Dict) -> float:
        """Assess reusability of monitoring patterns"""
        score = 8.0  # Monitoring is generally reusable
        
        if metadata.get('has_emojis', False):
            score += 1.0
        
        return min(10.0, score)


class DocumentationExtractor(ComponentExtractor):
    """Extracts documentation and comments"""
    
    def __init__(self):
        super().__init__()
        self.component_type = ComponentType.DOCUMENTATION
    
    def extract(self, code: str, source_generator: str = "unknown") -> List[ExtractedComponent]:
        """Extract documentation from code"""
        components = []
        
        # Extract section headers
        section_pattern = r'^# (📦|🚀|📊|🔍).*?SECTION.*?$'
        section_matches = re.finditer(section_pattern, code, re.MULTILINE)
        
        section_headers = []
        for match in section_matches:
            section_headers.append(match.group(0))
        
        if section_headers:
            content = '\n'.join(section_headers)
            metadata = {
                'section_count': len(section_headers),
                'has_emojis': True,
                'section_types': [h.split()[1] if len(h.split()) > 1 else 'unknown' for h in section_headers]
            }
            
            component = ExtractedComponent(
                component_type=self.component_type,
                content=content,
                quality_score=self._assess_quality(content, metadata),
                reusability_score=self._assess_reusability(content, metadata),
                source_generator=source_generator,
                extraction_confidence=0.9,
                dependencies=[],
                metadata=metadata
            )
            components.append(component)
        
        # Extract inline comments
        comment_pattern = r'#[^#].*$'
        comment_matches = re.finditer(comment_pattern, code, re.MULTILINE)
        
        inline_comments = []
        for match in comment_matches:
            comment = match.group(0).strip()
            if not any(emoji in comment for emoji in ['📦', '🚀', '📊', '🔍']):  # Skip section headers
                inline_comments.append(comment)
        
        if inline_comments and len(inline_comments) >= 3:  # Only extract if significant amount
            content = '\n'.join(inline_comments[:10])  # Limit to top 10
            metadata = {
                'comment_count': len(inline_comments),
                'avg_length': sum(len(c) for c in inline_comments) / len(inline_comments),
                'explanatory_comments': [c for c in inline_comments if len(c) > 30]
            }
            
            component = ExtractedComponent(
                component_type=self.component_type,
                content=content,
                quality_score=self._assess_quality(content, metadata),
                reusability_score=self._assess_reusability(content, metadata),
                source_generator=source_generator,
                extraction_confidence=0.7,
                dependencies=[],
                metadata=metadata
            )
            components.append(component)
        
        return components
    
    def _assess_quality(self, content: str, metadata: Dict) -> float:
        """Assess quality of documentation"""
        score = 5.0
        
        # Section headers are high quality
        if metadata.get('has_emojis', False):
            score += 2.0
        
        # Good balance of comments
        comment_count = metadata.get('comment_count', 0)
        if 3 <= comment_count <= 15:
            score += 1.5
        elif comment_count > 20:
            score -= 1.0
        
        # Explanatory comments are valuable
        explanatory_count = len(metadata.get('explanatory_comments', []))
        if explanatory_count > 0:
            score += min(2.0, explanatory_count * 0.3)
        
        return min(10.0, max(1.0, score))
    
    def _assess_reusability(self, content: str, metadata: Dict) -> float:
        """Assess reusability of documentation"""
        score = 6.0
        
        # Section headers are highly reusable
        if metadata.get('has_emojis', False):
            score += 2.0
        
        # General comments are moderately reusable
        if metadata.get('comment_count', 0) > 0:
            score += 1.0
        
        return min(10.0, score)


class DataRobotPatternExtractor(ComponentExtractor):
    """Extracts DataRobot-specific patterns and best practices"""
    
    def __init__(self):
        super().__init__()
        self.component_type = ComponentType.DATAROBOT_PATTERNS
    
    def extract(self, code: str, source_generator: str = "unknown") -> List[ExtractedComponent]:
        """Extract DataRobot patterns from code"""
        components = []
        
        # Client initialization pattern
        client_pattern = r'dr\.Client\([^)]*\)'
        client_matches = re.finditer(client_pattern, code)
        
        for match in client_matches:
            pattern = match.group(0)
            metadata = {
                'pattern_type': 'client_init',
                'has_token': 'token' in pattern,
                'has_endpoint': 'endpoint' in pattern
            }
            
            component = ExtractedComponent(
                component_type=self.component_type,
                content=pattern,
                quality_score=self._assess_quality(pattern, metadata),
                reusability_score=self._assess_reusability(pattern, metadata),
                source_generator=source_generator,
                extraction_confidence=0.95,
                dependencies=['imports'],
                metadata=metadata
            )
            components.append(component)
        
        # Project creation pattern
        project_pattern = r'dr\.Project\.create\([^)]*(?:\([^)]*\)[^)]*)*\)'
        project_matches = re.finditer(project_pattern, code, re.DOTALL)
        
        for match in project_matches:
            pattern = match.group(0)
            metadata = {
                'pattern_type': 'project_create',
                'has_sourcedata': 'sourcedata' in pattern,
                'has_project_name': 'project_name' in pattern
            }
            
            component = ExtractedComponent(
                component_type=self.component_type,
                content=pattern,
                quality_score=self._assess_quality(pattern, metadata),
                reusability_score=self._assess_reusability(pattern, metadata),
                source_generator=source_generator,
                extraction_confidence=0.9,
                dependencies=['imports'],
                metadata=metadata
            )
            components.append(component)
        
        # Autopilot patterns
        autopilot_patterns = [
            r'project\.set_target\([^)]*\)',
            r'project\.wait_for_autopilot\(\)',
            r'project\.get_models\(\)'
        ]
        
        for pattern_regex in autopilot_patterns:
            matches = re.finditer(pattern_regex, code)
            for match in matches:
                pattern = match.group(0)
                pattern_name = pattern.split('(')[0].split('.')[-1]
                
                metadata = {
                    'pattern_type': f'autopilot_{pattern_name}',
                    'method_name': pattern_name
                }
                
                component = ExtractedComponent(
                    component_type=self.component_type,
                    content=pattern,
                    quality_score=self._assess_quality(pattern, metadata),
                    reusability_score=self._assess_reusability(pattern, metadata),
                    source_generator=source_generator,
                    extraction_confidence=0.9,
                    dependencies=['imports'],
                    metadata=metadata
                )
                components.append(component)
        
        return components
    
    def _assess_quality(self, content: str, metadata: Dict) -> float:
        """Assess quality of DataRobot patterns"""
        score = 8.0  # DataRobot patterns are generally high quality
        
        pattern_type = metadata.get('pattern_type', '')
        
        if pattern_type == 'client_init':
            if metadata.get('has_token', False) and metadata.get('has_endpoint', False):
                score += 1.0
        elif pattern_type == 'project_create':
            if metadata.get('has_sourcedata', False) and metadata.get('has_project_name', False):
                score += 1.0
        
        return min(10.0, score)
    
    def _assess_reusability(self, content: str, metadata: Dict) -> float:
        """Assess reusability of DataRobot patterns"""
        return 9.0  # DataRobot patterns are highly reusable


class ComponentExtractionOrchestrator:
    """Orchestrates extraction from multiple specialized extractors"""
    
    def __init__(self):
        self.extractors = {
            ComponentType.IMPORTS: ImportExtractor(),
            ComponentType.ERROR_HANDLING: ErrorHandlingExtractor(),
            ComponentType.MONITORING: MonitoringExtractor(),
            ComponentType.DOCUMENTATION: DocumentationExtractor(),
            ComponentType.DATAROBOT_PATTERNS: DataRobotPatternExtractor()
        }
    
    def extract_all_components(self, code: str, source_generator: str = "unknown") -> Dict[ComponentType, List[ExtractedComponent]]:
        """Extract all types of components from code"""
        
        all_components = {}
        
        for component_type, extractor in self.extractors.items():
            try:
                components = extractor.extract(code, source_generator)
                if components:
                    all_components[component_type] = components
            except Exception as e:
                # Log error but continue with other extractors
                print(f"Warning: Failed to extract {component_type.value} from {source_generator}: {e}")
                continue
        
        return all_components
    
    def get_best_components(self, all_extractions: Dict[str, Dict[ComponentType, List[ExtractedComponent]]], 
                          component_type: ComponentType, 
                          min_quality: float = 6.0) -> List[ExtractedComponent]:
        """Get the best components of a specific type across all sources"""
        
        candidates = []
        
        for source_generator, extractions in all_extractions.items():
            if component_type in extractions:
                candidates.extend(extractions[component_type])
        
        # Filter by minimum quality and sort by combined quality + reusability
        quality_components = [c for c in candidates if c.quality_score >= min_quality]
        quality_components.sort(key=lambda c: (c.quality_score + c.reusability_score) / 2, reverse=True)
        
        return quality_components
    
    def create_synthesis_recommendations(self, all_extractions: Dict[str, Dict[ComponentType, List[ExtractedComponent]]]) -> Dict[str, List[str]]:
        """Create recommendations for which components to synthesize"""
        
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        for component_type in ComponentType:
            best_components = self.get_best_components(all_extractions, component_type)
            
            if not best_components:
                continue
                
            best_component = best_components[0]
            avg_score = (best_component.quality_score + best_component.reusability_score) / 2
            
            recommendation = f"{component_type.value} from {best_component.source_generator} (score: {avg_score:.1f})"
            
            if avg_score >= 8.0:
                recommendations['high_priority'].append(recommendation)
            elif avg_score >= 6.0:
                recommendations['medium_priority'].append(recommendation)
            else:
                recommendations['low_priority'].append(recommendation)
        
        return recommendations