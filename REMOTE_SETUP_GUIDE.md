# 🚀 Remote Environment Setup Guide

## Quick Setup Instructions

Now that the fixes have been pushed, here's how to get it working in your remote environment:

### 1. Fresh Clone (if needed)
```bash
cd ~/storage
rm -rf DR_API_CREWAI_AGENT  # Remove old clone if present
git clone https://github.com/jeremy-pernicek/DR_API_CREWAI_AGENT.git
cd DR_API_CREWAI_AGENT
```

### 2. Automated Installation
```bash
# Run the automated installer
python3 install.py
```

This will:
- Check and install all Python dependencies
- Verify knowledge base files are present
- Test basic imports
- Check API key configuration

### 3. Manual Installation (if needed)
```bash
# Install all dependencies at once
pip install -r requirements.txt

# Or install individually if some fail
pip install openai anthropic google-generativeai datarobot
pip install faiss-cpu rank_bm25 scikit-learn sentence-transformers
pip install pandas numpy requests nltk PyYAML beautifulsoup4 tqdm
```

### 4. Set API Keys (Optional for testing)
```bash
# For full functionality, set these:
export ANTHROPIC_API_KEY="your_claude_key"
export AZURE_OPENAI_API_KEY="your_azure_openai_key"
export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
export GOOGLE_API_KEY="your_gemini_key"
export DATAROBOT_API_TOKEN="your_datarobot_token"
```

### 5. Test the System
```bash
# Test components (no API calls needed)
python3 examples/system_test.py

# Generate code (requires API keys)
python3 examples/quick_generate.py "create a classification model"

# Interactive session (requires API keys)
python3 examples/interactive_session.py
```

## What Was Fixed

### ✅ **Knowledge Base Files**
- Added .pkl and .faiss files to git repository
- Updated .gitignore to include necessary index files
- Now includes ~10 MB of vector indexes and search data

### ✅ **Dependency Management** 
- Created install.py script for automated setup
- Updated requirements.txt with all dependencies
- Added fallback handling for missing packages

### ✅ **Better Error Handling**
- Graceful handling of missing vector stores
- Clear error messages for missing dependencies
- Fallback modes for various failure scenarios

## Expected Results

After running `python3 install.py`, you should see:

```
🚀 DataRobot AI Code Generation Agent - Installation
============================================================

📦 Checking Python dependencies...
✅ openai
✅ anthropic
✅ google-generativeai
✅ datarobot
✅ faiss-cpu
✅ rank_bm25
[... all dependencies listed ...]

📚 Checking knowledge base...
✅ Knowledge base found
   📊 45 index files
   📄 23 document files

🧪 Testing core imports...
✅ Core system imports working

🔑 Checking API configuration...
⚠️  Claude API - not configured
⚠️  Azure OpenAI API - not configured
⚠️  Gemini API - not configured
⚠️  DataRobot API - not configured

📊 Summary:
   Dependencies: ✅ 15/15 installed
   Knowledge Base: ✅
   API Keys: ⚠️  0/4 configured

⚠️  Setup incomplete. Please:
   • Configure API keys (see README.md)
```

## Testing Without API Keys

The system_test.py will work without API keys:

```bash
python3 examples/system_test.py
```

Expected output:
```
🚀 DataRobot Enhanced System Component Tests
============================================================

==================== Import Test ====================
🧪 Testing Imports...
✅ Shared models imported
✅ Code synthesizer imported
✅ Synthesis orchestrator imported
✅ Component extractors imported
✅ Enhanced SimpleMultiLLM imported
✅ PASSED

[... all tests should pass ...]

Overall: 4/4 tests passed
🎉 All components working correctly!
```

Let me know if you encounter any issues after pulling the latest changes!