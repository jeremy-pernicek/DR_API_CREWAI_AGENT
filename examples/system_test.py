#!/usr/bin/env python3
"""
DataRobot Agent System Component Test

This script tests all system components without making API calls to ensure
everything is properly configured and working.

Usage: python3 examples/system_test.py
"""

import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing Imports...")
    
    try:
        from agent_crewai.src.agents.shared_models import CodeSolution, GenerationMetrics, SynthesisStrategy
        print("✅ Shared models imported")
        
        from agent_crewai.src.agents.code_synthesizer import DataRobotCodeSynthesizer
        print("✅ Code synthesizer imported")
        
        from agent_crewai.src.agents.synthesis_orchestrator import SynthesisOrchestrator
        print("✅ Synthesis orchestrator imported")
        
        from agent_crewai.src.agents.component_extractors import ComponentExtractionOrchestrator
        print("✅ Component extractors imported")
        
        from agent_crewai.src.agents.simple_multi_llm import SimpleMultiLLM
        print("✅ Enhanced SimpleMultiLLM imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_synthesis_system():
    """Test the synthesis system with mock data"""
    print("\n🧪 Testing Synthesis System...")
    
    try:
        from agent_crewai.src.agents.shared_models import CodeSolution, SynthesisStrategy
        from agent_crewai.src.agents.code_synthesizer import DataRobotCodeSynthesizer
        from agent_crewai.src.agents.synthesis_orchestrator import SynthesisOrchestrator
        
        # Create mock solutions
        solutions = [
            CodeSolution(
                generator="Claude Sonnet 4 (Production-Ready)",
                code="import datarobot as dr\ndr.Client()\nproject = dr.Project.create(sourcedata='data.csv')",
                explanation="Simple project creation",
                score=7.5,
                generation_time=2.1
            ),
            CodeSolution(
                generator="GPT-4o (Enterprise-Grade)",
                code="import datarobot as dr\ntry:\n    dr.Client()\n    project = dr.Project.create(sourcedata='data.csv')\nexcept Exception as e:\n    print(f'Error: {e}')",
                explanation="Project creation with error handling",
                score=8.2,
                generation_time=3.2
            ),
            CodeSolution(
                generator="Gemini 2.5 Pro (Analytical)",
                code="import datarobot as dr\nimport pandas as pd\ndr.Client()\nprint('✅ Connected')\nproject = dr.Project.create(sourcedata='data.csv')\nprint('✅ Project created')",
                explanation="Project creation with progress indicators",
                score=7.8,
                generation_time=2.8
            )
        ]
        
        # Test synthesizer
        synthesizer = DataRobotCodeSynthesizer()
        print("✅ Synthesizer created")
        
        # Test strategy decision
        strategy, reasoning = synthesizer.decide_strategy(solutions)
        print(f"✅ Strategy decided: {strategy.value}")
        print(f"   Reasoning: {reasoning}")
        
        # Test solution analysis
        analyses = synthesizer.analyze_solutions(solutions)
        print(f"✅ Solutions analyzed: {len(analyses)} analyses created")
        
        # Test synthesis
        synthesis_result = synthesizer.synthesize_components(analyses, strategy)
        print(f"✅ Synthesis completed: {synthesis_result.strategy_used.value}")
        print(f"   Final score estimate: {synthesis_result.final_score_estimate}")
        print(f"   Source contributions: {len(synthesis_result.source_contributions)} sources")
        
        # Test orchestrator
        orchestrator = SynthesisOrchestrator(synthesizer, verbose=False)
        print("✅ Orchestrator created")
        
        final_result, metrics = orchestrator.orchestrate_synthesis(solutions)
        print(f"✅ Orchestration completed in {metrics.total_synthesis_time:.2f}s")
        print(f"   Quality improvement: +{metrics.quality_improvement:.1f}")
        print(f"   Final confidence: {metrics.final_confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Synthesis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_extraction():
    """Test component extraction system"""
    print("\n🧪 Testing Component Extraction...")
    
    try:
        from agent_crewai.src.agents.component_extractors import ComponentExtractionOrchestrator
        
        # Mock DataRobot code
        test_code = """
# 📦 SECTION 1: Quick Setup
import datarobot as dr
import pandas as pd

# Connect to DataRobot
dr.Client(token='YOUR_TOKEN', endpoint='https://app.datarobot.com/api/v2')
print("✅ Connected to DataRobot")

# 🚀 SECTION 2: Main Workflow
try:
    project = dr.Project.create(sourcedata='data.csv', project_name='My Project')
    print("✅ Project created")
    
    project.set_target(target='sales', mode=dr.AUTOPILOT_MODE.QUICK)
    project.wait_for_autopilot()
    print("🎉 Autopilot completed")
    
except Exception as e:
    print(f"❌ Error: {e}")

# 📊 SECTION 3: Results
models = project.get_models()
print(f"Found {len(models)} models")
"""
        
        extractor = ComponentExtractionOrchestrator()
        print("✅ Component extractor created")
        
        components = extractor.extract_all_components(test_code, "Test Generator")
        print(f"✅ Components extracted: {len(components)} types found")
        
        for component_type, component_list in components.items():
            print(f"   {component_type.value}: {len(component_list)} components")
            
        # Test recommendations
        all_extractions = {"test": components}
        recommendations = extractor.create_synthesis_recommendations(all_extractions)
        
        total_recs = sum(len(recs) for recs in recommendations.values())
        print(f"✅ Synthesis recommendations: {total_recs} total")
        
        return True
        
    except Exception as e:
        print(f"❌ Component extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_system_init():
    """Test that the enhanced system can initialize"""
    print("\n🧪 Testing Enhanced System Initialization...")
    
    try:
        from agent_crewai.src.agents.simple_multi_llm import SimpleMultiLLM
        
        # Initialize without making API calls
        enhanced_llm = SimpleMultiLLM(
            data_dir="agent_crewai/data",
            init_dr_client=False,  # Don't initialize DataRobot client
            use_practical_prompts=True
        )
        
        print("✅ Enhanced SimpleMultiLLM initialized")
        print(f"   Synthesis system: {'✅' if hasattr(enhanced_llm, 'code_synthesizer') else '❌'}")
        print(f"   Orchestrator: {'✅' if hasattr(enhanced_llm, 'synthesis_orchestrator') else '❌'}")
        print(f"   Component extractor: {'✅' if hasattr(enhanced_llm, 'component_extractor') else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced system init failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 DataRobot Enhanced System Component Tests")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Synthesis System Test", test_synthesis_system), 
        ("Component Extraction Test", test_component_extraction),
        ("Enhanced System Init Test", test_enhanced_system_init)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        print(f"{'✅ PASSED' if success else '❌ FAILED'}")
    
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<8} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All components working correctly!")
        print("🚀 Ready for full system testing with API calls")
    else:
        print("⚠️  Some components need fixes before full testing")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)