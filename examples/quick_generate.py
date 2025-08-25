#!/usr/bin/env python3
"""
Quick DataRobot Code Generation

Usage: python3 examples/quick_generate.py "your prompt here"

Example:
python3 examples/quick_generate.py "create a time series model to forecast sales"
"""

import os
import sys
import warnings
from pathlib import Path

# Setup
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress urllib3 OpenSSL warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')

def clean_api_token():
    """Clean the API token of any Unicode issues"""
    token = os.getenv('DATAROBOT_API_TOKEN')
    if token:
        clean_token = token.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        clean_token = clean_token.encode('ascii', errors='ignore').decode('ascii')
        os.environ['DATAROBOT_API_TOKEN'] = clean_token
        return clean_token
    return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 examples/quick_generate.py \"your prompt here\"")
        print("\nExample:")
        print("python3 examples/quick_generate.py \"create a time series model to forecast sales\"")
        sys.exit(1)
    
    prompt = sys.argv[1]
    print(f"🎯 Prompt: {prompt}")
    print("=" * 80)
    
    # Clean API token
    clean_api_token()
    
    try:
        from agent_crewai.src.agents.simple_multi_llm import SimpleMultiLLM
        
        # Initialize the enhanced system
        llm = SimpleMultiLLM(
            data_dir="agent_crewai/data",
            init_dr_client=True,
            use_practical_prompts=True
        )
        
        # Generate code
        result = llm.generate_code(prompt, enable_synthesis=True)
        
        if result['status'] == 'success':
            best_sol = result['best_solution']
            
            print(f"\n{'='*80}")
            print(f"📋 COMPLETE GENERATED CODE (Copy & Paste Ready)")
            print(f"{'='*80}")
            
            # Print code without line numbers for easy copy-paste
            print(best_sol['code'])
            
            print(f"{'='*80}")
            
            print(f"\n🎉 SUCCESS!")
            print(f"🏆 Solution: {best_sol['generator']}")
            print(f"📊 Score: {best_sol['score']}/10")
            print(f"⏱️  Total Time: {result['metrics']['total_time']:.2f}s")
            
        else:
            print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()