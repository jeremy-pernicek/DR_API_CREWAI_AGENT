"""
DataRobot API Code Generation Assistant - Custom Model Entry Point

This is the main entry point for deploying the assistant as a DataRobot custom model.
It orchestrates multi-LLM code generation using the DataRobot LLM Gateway.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_crewai.src.rag.hybrid_retriever import HybridRetriever, create_sample_data
from agent_crewai.src.agents.simple_multi_llm import SimpleMultiLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataRobotCodeAssistant:
    """
    DataRobot API Code Generation Assistant
    
    A multi-LLM orchestration system that generates DataRobot Python SDK code
    from natural language requests using hybrid retrieval and quality scoring.
    """
    
    def __init__(self):
        """Initialize the assistant"""
        logger.info("Initializing DataRobot API Code Generation Assistant...")
        
        try:
            # Initialize retriever system
            self.retriever = self._setup_retriever()
            
            # Initialize multi-LLM system
            self.multi_llm = SimpleMultiLLM(retriever=self.retriever)
            
            logger.info("✅ Assistant initialized successfully")
            logger.info(f"📊 Retriever stats: {self.retriever.get_stats()}")
            logger.info(f"🤖 LLM models: {list(self.multi_llm.models.values())}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize assistant: {str(e)}")
            raise
    
    def _setup_retriever(self) -> HybridRetriever:
        """Set up the hybrid retrieval system"""
        logger.info("Setting up hybrid retrieval system...")
        
        retriever = HybridRetriever(
            data_dir="agent_crewai/data",
            bm25_weight=0.6,
            semantic_weight=0.4
        )
        
        # Check if we have existing indexes
        if retriever.documents:
            logger.info(f"Loaded existing indexes with {len(retriever.documents)} documents")
        else:
            logger.info("No existing indexes found, creating sample documentation...")
            
            # Create sample DataRobot documentation
            sample_docs = create_sample_data()
            
            # Add additional comprehensive examples
            from agent_crewai.src.rag.hybrid_retriever import Document
            
            additional_docs = [
                Document(
                    id="time_series_complete",
                    title="Complete Time Series Project Setup",
                    content="""Complete time series project setup requires: 1) datetime_partition_column for time-aware splits, 2) forecast_window_start and forecast_window_end for prediction horizon, 3) feature_derivation_window_start/end for historical lookback, 4) series_id_columns for multiseries, 5) known_in_advance features for future data. Use project.set_target() with partitioning_method=DatetimePartitioningSpecification.""",
                    type="comprehensive_example",
                    source="python_sdk",
                    url="https://docs.datarobot.com/timeseries-complete"
                ),
                Document(
                    id="deployment_scoring",
                    title="Model Deployment and Scoring Pipeline",
                    content="""Deploy models with dr.Deployment.create_from_learning_model(). For batch scoring: use BatchPredictionJob.score() with intake_settings for data source, output_settings for results. Configure prediction_explanations_enabled=True for model insights. For time series: set timeseries_settings with forecast_point and series_id columns.""",
                    type="comprehensive_example", 
                    source="python_sdk",
                    url="https://docs.datarobot.com/deployment-scoring"
                ),
                Document(
                    id="error_handling_patterns",
                    title="DataRobot SDK Error Handling Best Practices",
                    content="""Handle DataRobot errors with try/except blocks: ClientError for API issues, AsyncProcessUnsuccessfulError for failed async operations, JobFailedException for failed jobs. Always check job status with wait_for_completion(). Include proper logging and cleanup on failures. Use exponential backoff for retries.""",
                    type="best_practice",
                    source="python_sdk", 
                    url="https://docs.datarobot.com/error-handling"
                )
            ]
            
            all_docs = sample_docs + additional_docs
            retriever.add_documents(all_docs)
            retriever.build_indexes()
            
            logger.info(f"Built indexes for {len(all_docs)} documents")
        
        return retriever
    
    def generate_code(self, user_request: str) -> Dict[str, Any]:
        """
        Generate DataRobot API code from natural language request
        
        Args:
            user_request: Natural language description of what to build
            
        Returns:
            Dict containing generated code and metadata
        """
        logger.info(f"🔍 Processing request: {user_request}")
        
        try:
            result = self.multi_llm.generate_code(user_request)
            
            if result['status'] == 'success':
                logger.info(f"✅ Generated code successfully using {result['best_solution']['generator']}")
                logger.info(f"📊 Quality score: {result['best_solution']['score']}/10")
            else:
                logger.error(f"❌ Code generation failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Exception during code generation: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and capabilities"""
        return {
            'name': 'DataRobot API Code Generation Assistant',
            'version': '1.0.0',
            'description': 'Multi-LLM orchestration system for DataRobot Python SDK code generation',
            'capabilities': [
                'DataRobot project creation and configuration',
                'Time series modeling setup', 
                'Model deployment and scoring',
                'Batch prediction workflows',
                'Error handling and best practices',
                'Multi-LLM code generation and quality scoring'
            ],
            'models_available': list(self.multi_llm.models.values()),
            'retriever_stats': self.retriever.get_stats(),
            'supported_scenarios': [
                'Basic project creation from CSV',
                'Time series forecasting projects',
                'Model deployment and endpoints',
                'Batch prediction jobs',
                'Advanced configurations and workflows'
            ]
        }


# Global assistant instance (initialized once)
_assistant = None

def get_assistant() -> DataRobotCodeAssistant:
    """Get or create the global assistant instance"""
    global _assistant
    if _assistant is None:
        _assistant = DataRobotCodeAssistant()
    return _assistant


# DataRobot Custom Model Interface Functions
def init(code_dir: str, **kwargs) -> None:
    """Initialize the custom model (DataRobot entry point)"""
    logger.info("🚀 Initializing DataRobot API Code Generation Assistant...")
    
    # Initialize the assistant
    get_assistant()
    
    logger.info("✅ Custom model initialized successfully")


def predict(data: str, model: Any = None, **kwargs) -> str:
    """
    Generate DataRobot API code (DataRobot prediction entry point)
    
    Args:
        data: User request as string
        
    Returns:
        JSON string containing generated code and metadata
    """
    assistant = get_assistant()
    
    try:
        # Parse input data
        if isinstance(data, str):
            user_request = data.strip()
        else:
            user_request = str(data)
        
        logger.info(f"Processing prediction request: {user_request[:100]}...")
        
        # Generate code
        result = assistant.generate_code(user_request)
        
        # Return JSON result
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        error_result = {
            'status': 'error',
            'error': str(e),
            'input': user_request if 'user_request' in locals() else str(data)
        }
        return json.dumps(error_result, indent=2)


# Command Line Interface
def main():
    """Command line interface for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DataRobot API Code Generation Assistant")
    parser.add_argument("--request", "-r", type=str, required=True, help="Natural language request")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize assistant
    assistant = get_assistant()
    
    if args.info:
        info = assistant.get_system_info()
        print(json.dumps(info, indent=2))
        return
    
    # Process request
    print(f"🔍 Request: {args.request}")
    print("=" * 80)
    
    result = assistant.generate_code(args.request)
    
    if result['status'] == 'success':
        best = result['best_solution']
        print(f"✅ Generated by {best['generator']} (Score: {best['score']}/10)")
        print("\n📝 Generated Code:")
        print("-" * 40)
        print(best['code'])
        print("-" * 40)
        
        print(f"\n📊 Solution Comparison:")
        for sol in result.get('all_solutions', []):
            print(f"  - {sol['generator']}: {sol['score']}/10")
    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    main()