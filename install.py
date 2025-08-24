#!/usr/bin/env python3
"""
DataRobot AI Code Generation Agent Installation Script

This script ensures all dependencies are installed and the knowledge base is ready.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package):
    """Check if a package is installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    print("🚀 DataRobot AI Code Generation Agent - Installation")
    print("=" * 60)
    
    # Core requirements with fallbacks
    requirements = [
        ("openai", "openai"),
        ("anthropic", "anthropic"), 
        ("google-generativeai", "google.generativeai"),
        ("datarobot", "datarobot"),
        ("faiss-cpu", "faiss"),
        ("rank_bm25", "rank_bm25"),
        ("scikit-learn", "sklearn"),
        ("sentence-transformers", "sentence_transformers"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("requests", "requests"),
        ("nltk", "nltk"),
        ("PyYAML", "yaml"),
        ("beautifulsoup4", "bs4"),
        ("tqdm", "tqdm")
    ]
    
    print("\n📦 Checking Python dependencies...")
    
    missing = []
    for package, import_name in requirements:
        if check_package(import_name):
            print(f"✅ {package}")
        else:
            print(f"⚠️  {package} - installing...")
            if install_package(package):
                print(f"✅ {package} - installed successfully")
            else:
                print(f"❌ {package} - failed to install")
                missing.append(package)
    
    if missing:
        print(f"\n❌ Failed to install: {', '.join(missing)}")
        print("Please install manually with:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print("")
    
    # Check knowledge base
    print("\n📚 Checking knowledge base...")
    
    project_root = Path(__file__).parent
    indexes_dir = project_root / "agent_crewai" / "data" / "indexes"
    scraped_dir = project_root / "agent_crewai" / "data" / "scraped"
    
    if indexes_dir.exists() and scraped_dir.exists():
        # Count files
        index_files = list(indexes_dir.rglob("*"))
        scraped_files = list(scraped_dir.glob("*.json"))
        
        print(f"✅ Knowledge base found")
        print(f"   📊 {len(index_files)} index files")
        print(f"   📄 {len(scraped_files)} document files")
        
        # Check for key files
        key_files = [
            "embeddings.npy",
            "semantic_index.faiss", 
            "documents.json",
            "bm25_index.pkl"
        ]
        
        missing_files = []
        for key_file in key_files:
            if not any(f.name == key_file for f in index_files):
                missing_files.append(key_file)
        
        if missing_files:
            print(f"⚠️  Missing key files: {', '.join(missing_files)}")
            print("   Knowledge base may not work optimally")
        
    else:
        print("❌ Knowledge base not found")
        print("   The system will attempt to create minimal indexes on first run")
    
    # Test basic imports
    print("\n🧪 Testing core imports...")
    
    try:
        from agent_crewai.src.agents.simple_multi_llm import SimpleMultiLLM
        print("✅ Core system imports working")
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    # Check API keys
    print("\n🔑 Checking API configuration...")
    
    api_keys = [
        ("ANTHROPIC_API_KEY", "Claude API"),
        ("AZURE_OPENAI_API_KEY", "Azure OpenAI API"),
        ("GOOGLE_API_KEY", "Gemini API"), 
        ("DATAROBOT_API_TOKEN", "DataRobot API")
    ]
    
    configured = 0
    for env_var, description in api_keys:
        if os.getenv(env_var):
            print(f"✅ {description}")
            configured += 1
        else:
            print(f"⚠️  {description} - not configured")
    
    print(f"\n📊 Summary:")
    print(f"   Dependencies: {'✅' if not missing else '⚠️ '} {len(requirements) - len(missing)}/{len(requirements)} installed")
    print(f"   Knowledge Base: {'✅' if indexes_dir.exists() else '❌'}")
    print(f"   API Keys: {'✅' if configured >= 2 else '⚠️ '} {configured}/{len(api_keys)} configured")
    
    if not missing and indexes_dir.exists() and configured >= 2:
        print(f"\n🎉 Installation complete! Ready to generate code.")
        print(f"\nTry running:")
        print(f"  python3 examples/system_test.py")
        print(f"  python3 examples/quick_generate.py \"create a time series model\"")
        return True
    else:
        print(f"\n⚠️  Setup incomplete. Please:")
        if missing:
            print(f"   • Install missing packages: {', '.join(missing)}")
        if not indexes_dir.exists():
            print(f"   • Knowledge base will be created on first run")
        if configured < 2:
            print(f"   • Configure API keys (see README.md)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)