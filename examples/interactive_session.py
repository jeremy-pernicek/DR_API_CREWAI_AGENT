#!/usr/bin/env python3
"""
Interactive DataRobot Code Generation Session

This script provides an interactive session where you can:
- Generate DataRobot code from prompts
- Request revisions and improvements
- Save code to files
- Generate multiple variations

Usage: python3 examples/interactive_session.py
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

def print_full_code(code, title="Generated Code"):
    """Print the complete code with clear formatting"""
    print(f"\n{'='*80}")
    print(f"📋 {title.upper()}")
    print(f"{'='*80}")
    
    # Print code without line numbers for easy copy-paste
    print(code)
    
    print(f"{'='*80}\n")

def save_code(code, filename):
    """Save code to file and return the path"""
    try:
        with open(filename, 'w') as f:
            f.write(code)
        return filename
    except Exception as e:
        print(f"⚠️  Could not save to {filename}: {e}")
        return None

def generate_code(llm, prompt, enable_synthesis=True):
    """Generate code and return the result"""
    print(f"\n🚀 Generating code for: {prompt}")
    print("-" * 60)
    
    try:
        result = llm.generate_code(prompt, enable_synthesis=enable_synthesis)
        
        if result['status'] == 'success':
            return result
        else:
            print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"💥 Error during generation: {e}")
        return None

def process_revision_request(llm, original_prompt, original_result, revision_request):
    """Process a revision request"""
    
    # Create enhanced prompt for revision
    revision_prompt = f"""REVISION REQUEST:

Original Request: {original_prompt}

Current Generated Code:
```python
{original_result['best_solution']['code']}
```

User's Revision Request: {revision_request}

Please generate an improved version of the code that addresses the user's feedback while maintaining all DataRobot best practices (use analyze_and_model(), configurable variables, no sample data).
"""
    
    print(f"\n🔄 Processing revision request...")
    return generate_code(llm, revision_prompt, enable_synthesis=True)

def interactive_session():
    """Run interactive code generation session"""
    print("🚀 Interactive DataRobot Code Generation Session")
    print("=" * 80)
    
    # Clean API token
    clean_api_token()
    
    try:
        from agent_crewai.src.agents.simple_multi_llm import SimpleMultiLLM
        
        llm = SimpleMultiLLM(
            data_dir="agent_crewai/data",
            init_dr_client=True,
            use_practical_prompts=True
        )
        print("\nSystem ready!")
        
        # Main generation loop
        while True:
            print("\n" + "="*80)
            print("📝 DATAROBOT CODE GENERATION")
            print("="*80)
            
            # Get user prompt
            print("\nWhat DataRobot task would you like help with?")
            print("(or 'exit' to quit)")
            user_prompt = input("> ").strip()
            
            if user_prompt.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_prompt:
                print("Please enter a prompt or 'exit' to quit.")
                continue
            
            # Generate initial code
            result = generate_code(llm, user_prompt)
            
            if not result:
                continue
            
            # Display results
            best_sol = result['best_solution']
            
            # Show full code
            print_full_code(best_sol['code'], "Your DataRobot Code")
            
            # Prepare filename for potential saving
            safe_prompt = user_prompt[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
            filename = f"generated_{safe_prompt}.py"
            
            # Next steps loop
            while True:
                print("\n" + "-"*60)
                print("🚀 NEXT STEPS")
                print("-"*60)
                print("1. Save code to file")
                print("2. Request a revision/improvement")
                print("3. Generate new code (different prompt)")
                print("4. Exit")
                
                choice = input("\nWhat would you like to do? (1-4): ").strip()
                
                if choice == '1':
                    # Save code to file
                    saved_path = save_code(best_sol['code'], filename)
                    if saved_path:
                        print(f"💾 Code saved to: {saved_path}")
                    else:
                        print("⚠️ Could not save file (may not be supported in this environment)")
                    
                elif choice == '2':
                    print("\nWhat changes would you like me to make?")
                    print("Examples:")
                    print("- 'Add error handling for file not found'")
                    print("- 'Include model evaluation metrics'") 
                    print("- 'Add time series specific parameters'")
                    print("- 'Make it work with PostgreSQL database'")
                    
                    revision_request = input("\nRevision request: ").strip()
                    
                    if revision_request:
                        revised_result = process_revision_request(llm, user_prompt, result, revision_request)
                        
                        if revised_result:
                            revised_sol = revised_result['best_solution']
                            print(f"\n🔄 Revision Complete!")
                            
                            print_full_code(revised_sol['code'], "Revised DataRobot Code")
                            
                            # Update current result for potential further revisions
                            result = revised_result
                            best_sol = revised_sol  # Update for potential saving
                            
                            # Update filename for revised version
                            filename = f"revised_{filename}"
                            
                        else:
                            print("❌ Revision failed. Please try a different request.")
                    
                elif choice == '3':
                    break  # Go back to main prompt
                    
                elif choice == '4':
                    print("👋 Goodbye!")
                    return
                    
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
                    
    except KeyboardInterrupt:
        print(f"\n\n🛑 Interrupted by user. Goodbye!")
        
    except Exception as e:
        print(f"\n💥 System Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    interactive_session()
    print("\n🏁 Session completed!")

if __name__ == "__main__":
    main()