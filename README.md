# 🚀 DataRobot AI Code Generation Agent

An intelligent multi-LLM code generation system that creates production-ready DataRobot API code. This agent leverages Claude, GPT-4o, and Gemini in parallel to generate high-quality, well-documented Python code for DataRobot machine learning workflows.

## 🌟 Key Features

### 🤖 **Multi-LLM Orchestration**
- **Parallel Processing**: Runs Claude Sonnet 4, GPT-4o, and Gemini 2.5 Pro simultaneously
- **Hybrid Synthesis**: Automatically combines the best components from multiple LLMs
- **Quality Scoring**: Real-time evaluation and ranking of generated solutions
- **Fallback Handling**: Robust error handling with backup models

### 🧬 **Advanced Code Synthesis** 
- **Component Extraction**: Identifies reusable code patterns across solutions
- **Intelligent Merging**: Combines best practices from multiple AI models
- **Quality Enhancement**: Iterative improvement of synthesized code
- **Best Practice Enforcement**: Ensures DataRobot API best practices

### 📚 **Comprehensive Knowledge Base**
- **360+ Documents**: DataRobot SDK docs, API references, community examples
- **Smart Retrieval**: Hybrid BM25 + semantic search for relevant context
- **Multi-Source**: GitHub repos, official docs, community notebooks
- **Always Updated**: Fresh examples and patterns

### ⚡ **Production-Ready Output**
- **Clean Interface**: Professional output without technical clutter
- **Copy-Paste Ready**: Numbered code with clear sections
- **Configurable Variables**: Easy customization for different use cases
- **Progress Indicators**: Real-time feedback on generation process

## 📁 Project Structure

```
DR_API_CREWAI_AGENT/
├── agent_crewai/
│   ├── src/
│   │   ├── agents/           # Core LLM agents and orchestration
│   │   │   ├── simple_multi_llm.py      # Main multi-LLM coordinator
│   │   │   ├── code_synthesizer.py      # Hybrid code synthesis
│   │   │   ├── synthesis_orchestrator.py # Synthesis pipeline
│   │   │   ├── component_extractors.py   # Code component analysis
│   │   │   └── shared_models.py          # Shared data structures
│   │   └── rag/              # Retrieval-Augmented Generation
│   │       ├── enhanced_hybrid_retriever.py # Context retrieval
│   │       ├── context_assembler.py         # Context assembly
│   │       └── content_processor.py         # Document processing
│   └── data/
│       ├── indexes/          # Search indexes (BM25 + semantic)
│       └── scraped/          # Knowledge base documents
├── examples/
│   ├── quick_generate.py     # Simple command-line generation
│   ├── interactive_session.py # Full interactive session
│   └── system_test.py        # Component testing
└── scripts/
    └── run_comprehensive_scrape.py # Knowledge base updates
```

## 🚀 Quick Start

### Prerequisites

1. **API Keys** - Set up your LLM provider API keys:
   ```bash
   export ANTHROPIC_API_KEY="your_claude_key"
   export AZURE_OPENAI_API_KEY="your_azure_key" 
   export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
   export GOOGLE_API_KEY="your_gemini_key"
   ```

2. **DataRobot API** - Configure DataRobot access:
   ```bash
   export DATAROBOT_API_TOKEN="your_dr_token"
   export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"
   ```

3. **Python Dependencies**:
   ```bash
   pip install datarobot anthropic openai google-generativeai
   pip install faiss-cpu scikit-learn numpy pandas rank_bm25
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/jeremy-pernicek/DR_API_CREWAI_AGENT.git
cd DR_API_CREWAI_AGENT

# Install dependencies
pip install -r requirements.txt

# Or run the automated installer
python3 install.py
```

**Note**: If you have issues with `faiss`, install `faiss-cpu` specifically:
```bash
pip install faiss-cpu
```

### Test the System

```bash
python3 examples/system_test.py
```

## 💻 Usage Examples

### 1. Quick Generation

Generate DataRobot code from a single prompt:

```bash
python3 examples/quick_generate.py "create a time series model to forecast sales"
```

**Output:**
```
🚀 Generating DataRobot code for: create a time series model to forecast sales
🔍 Phase 1: Context Assembly & Requirements Analysis
🤖 Phase 2: Parallel Multi-LLM Code Generation
  ✅ [1/3] GPT-4o (Enterprise-Grade) completed in 12.03s
  ✅ [2/3] Gemini 2.5 Pro (Analytical) completed in 36.35s
  ✅ [3/3] Claude Sonnet 4 (Production-Ready) completed in 1m 33.15s
📊 Phase 3: Solution Scoring & Evaluation
  📊 GPT-4o (Enterprise-Grade): 10.0/10 (PASS)
🧬 Phase 4: Hybrid Code Synthesis
✅ Final solution: GPT-4o (Enterprise-Grade) (Score: 10.0/10)

📋 COMPLETE GENERATED CODE (Copy & Paste Ready)
================================================================================
  1 | import datarobot as dr
  2 | import pandas as pd
  3 | 
  4 | # 📝 Configuration Variables (Edit These)
  5 | DATAROBOT_API_TOKEN = "INSERT_YOUR_API_TOKEN"
  6 | DATA_FILE_PATH = "sales_data.csv"
  7 | TARGET_COLUMN = "sales"
  8 | DATETIME_COLUMN = "date"
  9 | 
 10 | # Connect to DataRobot
 11 | dr.Client(token=DATAROBOT_API_TOKEN)
 12 | print("✅ Connected to DataRobot!")
 13 | 
 14 | # Load and prepare data
 15 | df = pd.read_csv(DATA_FILE_PATH)
 16 | df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
 17 | 
 18 | # Create time series project
 19 | project = dr.Project.analyze_and_model(
 20 |     source=df,
 21 |     target=TARGET_COLUMN,
 22 |     datetime_partition_column=DATETIME_COLUMN,
 23 |     forecast_window_start=1,
 24 |     forecast_window_end=7
 25 | )
 26 | 
 27 | print("🎉 Time series model created successfully!")
```

### 2. Interactive Session

Launch an interactive session with revision support:

```bash
python3 examples/interactive_session.py
```

Features:
- **Multi-turn conversations**: Generate, revise, and refine code
- **Save to file**: Export generated code 
- **Revision requests**: "Add error handling", "Include model metrics", etc.
- **Multiple attempts**: Try different approaches

### 3. Programmatic Usage

Use the system in your own Python code:

```python
from agent_crewai.src.agents.simple_multi_llm import SimpleMultiLLM

# Initialize the system
llm = SimpleMultiLLM(
    data_dir="agent_crewai/data",
    init_dr_client=True,
    use_practical_prompts=True
)

# Generate code
result = llm.generate_code(
    "create a binary classification model with feature importance analysis",
    enable_synthesis=True
)

if result['status'] == 'success':
    best_solution = result['best_solution']
    print(f"Generated by: {best_solution['generator']}")
    print(f"Quality score: {best_solution['score']}/10")
    print(f"Code:\n{best_solution['code']}")
```

## 🛠️ Advanced Features

### Multi-LLM Configuration

The system uses three specialized LLM profiles:

- **Claude Sonnet 4 (Production-Ready)**: Robust, reliable code with best practices
- **GPT-4o (Enterprise-Grade)**: High-quality, well-structured implementations  
- **Gemini 2.5 Pro (Analytical)**: Detailed analysis and comprehensive solutions

### Hybrid Synthesis Strategies

1. **Best Only**: Use the highest-scoring solution
2. **Enhanced**: Improve the best solution with components from others
3. **Hybrid**: Merge complementary strengths from multiple solutions
4. **Recovery**: Fallback mode for error handling

### Context Assembly

The system intelligently retrieves relevant context:

- **Query Analysis**: Understands intent and requirements
- **Multi-Source Search**: GitHub, docs, community examples
- **Relevance Ranking**: BM25 + semantic similarity
- **Context Optimization**: Balances completeness and relevance

## 🎯 Supported DataRobot Use Cases

- ✅ **Time Series Forecasting**: Sales, demand, financial predictions
- ✅ **Binary/Multi-class Classification**: Customer churn, fraud detection
- ✅ **Regression**: Price prediction, risk scoring
- ✅ **Anomaly Detection**: Outlier identification
- ✅ **Feature Engineering**: Automated feature creation
- ✅ **Model Deployment**: Prediction server setup
- ✅ **Batch Prediction**: Large-scale scoring
- ✅ **Model Management**: Lifecycle operations

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Provider Keys
ANTHROPIC_API_KEY=your_claude_key
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
GOOGLE_API_KEY=your_gemini_key

# DataRobot API
DATAROBOT_API_TOKEN=your_datarobot_token
DATAROBOT_ENDPOINT=https://app.datarobot.com/api/v2

# Optional: Model Configuration
CLAUDE_MODEL=claude-3-5-sonnet-20241022
GPT_MODEL=gpt-4o-2024-11-20
GEMINI_MODEL=gemini-2.0-flash-exp
```

### System Settings

The system can be configured for different environments:

```python
llm = SimpleMultiLLM(
    data_dir="agent_crewai/data",        # Knowledge base location
    init_dr_client=True,                  # Initialize DataRobot client
    use_practical_prompts=True,           # Use production prompts
    enable_synthesis=True,                # Enable hybrid synthesis
    timeout=300                           # API timeout (seconds)
)
```

## 🧪 Testing

### System Components Test

Verify all components work without API calls:

```bash
python3 examples/system_test.py
```

### Full Integration Test

Test with actual API calls:

```bash
python3 examples/quick_generate.py "create a simple classification model"
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `python3 examples/system_test.py`
5. **Commit**: `git commit -m "Add amazing feature"`
6. **Push**: `git push origin feature/amazing-feature`
7. **Create a Pull Request**

### Development Setup

```bash
git clone https://github.com/jeremy-pernicek/DR_API_CREWAI_AGENT.git
cd DR_API_CREWAI_AGENT

# Install development dependencies
pip install -e .
pip install pytest black isort

# Run tests
python3 examples/system_test.py
```

## 📚 Knowledge Base Updates

To update the knowledge base with latest DataRobot documentation:

```bash
python3 scripts/run_comprehensive_scrape.py
```

This will:
- Scrape latest DataRobot Python SDK docs
- Update community examples from GitHub
- Refresh API documentation
- Rebuild search indexes

## 🔧 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   Solution: Verify all API keys are set and valid
   Check: echo $ANTHROPIC_API_KEY
   ```

2. **Import Errors**  
   ```
   Solution: Ensure you're running from the project root
   Fix: python3 -m examples.quick_generate "your prompt"
   ```

3. **Empty Results**
   ```
   Solution: Check DataRobot API token and endpoint
   Test: python3 examples/system_test.py
   ```

4. **Slow Performance**
   ```
   Solution: Check network connection and API quotas
   Alternative: Use single LLM mode for faster results
   ```

### Debug Mode

Enable verbose output for debugging:

```python
llm = SimpleMultiLLM(
    data_dir="agent_crewai/data",
    init_dr_client=True,
    use_practical_prompts=True
)

# Enable debug output
llm.synthesis_orchestrator.set_quiet_mode(False)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DataRobot** - For the excellent Python SDK and documentation
- **Anthropic, OpenAI, Google** - For providing powerful LLM APIs
- **Community Contributors** - For DataRobot examples and best practices

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/jeremy-pernicek/DR_API_CREWAI_AGENT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jeremy-pernicek/DR_API_CREWAI_AGENT/discussions)
- **Documentation**: [Project Wiki](https://github.com/jeremy-pernicek/DR_API_CREWAI_AGENT/wiki)

---

**Made with ❤️ for the DataRobot community**

*Generate production-ready DataRobot code with the power of multiple AI models working together.*