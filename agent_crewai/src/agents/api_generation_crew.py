"""
Multi-LLM CrewAI implementation for DataRobot API code generation.
Uses DataRobot LLM Gateway for unified access to all models.
"""

import os
import datarobot as dr
from typing import Dict, List, Optional
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class DataRobotAPIGenerationCrew:
    """
    Orchestrates multiple LLMs for DataRobot API code generation via LLM Gateway.
    
    Architecture:
    - Planning: Gemini 2.0 Flash (fast planning and analysis)
    - Generation A: Claude 4.1 (highest accuracy for DataRobot APIs)  
    - Generation B: GPT-4o (robust code generation)
    - Quality Scoring: Gemini 2.0 Flash (fast evaluation)
    - Review: Claude 4.1 (final review)
    """
    
    def __init__(self, retriever=None):
        """Initialize the multi-LLM crew."""
        self.retriever = retriever
        self.dr_client = dr.Client()
        self.llm_base_url = f"{self.dr_client.endpoint}/genai/llmgw"
        self.dr_api_token = os.getenv('DATAROBOT_API_TOKEN')
        
        # Model configurations
        self.models = {
            'planner': 'vertex_ai/gemini-2.0-flash-001',
            'generator_a': 'bedrock/anthropic.claude-opus-4-1-20250805-v1:0', 
            'generator_b': 'azure/gpt-4o-2024-11-20',
            'scorer': 'vertex_ai/gemini-2.0-flash-001',
            'reviewer': 'bedrock/anthropic.claude-opus-4-1-20250805-v1:0'
        }
        
        self.crew = self._create_crew()
        
    def _create_llm_client(self, model_name: str, temperature: float = 0.1) -> ChatOpenAI:
        """Create LLM client for DataRobot Gateway"""
        return ChatOpenAI(
            base_url=self.llm_base_url,
            api_key=self.dr_api_token,
            model=model_name,
            temperature=temperature,
            max_tokens=4096
        )
        
    def _create_crew(self) -> Crew:
        """Create the specialized agent crew."""
        # Initialize LLM clients via DataRobot Gateway
        planner_llm = self._create_llm_client(self.models['planner'], temperature=0.0)
        generator_a_llm = self._create_llm_client(self.models['generator_a'], temperature=0.1)
        generator_b_llm = self._create_llm_client(self.models['generator_b'], temperature=0.1)
        scorer_llm = self._create_llm_client(self.models['scorer'], temperature=0.0)
        reviewer_llm = self._create_llm_client(self.models['reviewer'], temperature=0.1)
        
        # Create agents
        agents = self._create_agents(
            planner_llm, 
            generator_a_llm, 
            generator_b_llm,
            scorer_llm,
            reviewer_llm
        )
        
        # Create tasks
        tasks = self._create_tasks(agents)
        
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
    
    def _create_agents(self, planner_llm, generator_a_llm, generator_b_llm, scorer_llm, reviewer_llm):
        """Create specialized agents."""
        
        research_analyst = Agent(
            role='DataRobot Research Analyst',
            goal='Research and understand DataRobot API requirements from user requests',
            backstory="""You are an expert DataRobot consultant who analyzes user requirements 
            and identifies the specific DataRobot APIs, methods, and patterns needed. You have 
            deep knowledge of DataRobot's Python SDK, time series modeling, deployments, and 
            best practices.""",
            llm=planner_llm,
            verbose=True,
            allow_delegation=False
        )
        
        claude_generator = Agent(
            role='Claude DataRobot Code Generator',
            goal='Generate accurate, production-ready DataRobot Python SDK code',
            backstory="""You are Claude, an expert DataRobot developer specializing in the 
            DataRobot Python SDK. You write clean, efficient, and well-documented code that 
            follows DataRobot best practices. You excel at complex time series configurations, 
            deployment workflows, and error handling.""",
            llm=generator_a_llm,
            verbose=True,
            allow_delegation=False
        )
        
        gpt_generator = Agent(
            role='GPT DataRobot Code Generator', 
            goal='Generate robust DataRobot Python SDK code with comprehensive error handling',
            backstory="""You are GPT-4o, a DataRobot coding expert focused on creating robust, 
            enterprise-grade code. You excel at complex logic, comprehensive error handling, 
            logging, and creating modular, reusable code patterns for DataRobot workflows.""",
            llm=generator_b_llm,
            verbose=True,
            allow_delegation=False
        )
        
        quality_scorer = Agent(
            role='Code Quality Analyst',
            goal='Evaluate and score DataRobot code solutions for accuracy and best practices',
            backstory="""You are a DataRobot quality assurance expert who evaluates code 
            solutions. You assess accuracy, completeness, error handling, documentation, 
            adherence to DataRobot best practices, and overall code quality.""",
            llm=scorer_llm,
            verbose=True,
            allow_delegation=False
        )
        
        final_reviewer = Agent(
            role='Senior DataRobot Architect',
            goal='Provide final review and select the best code solution',
            backstory="""You are a senior DataRobot architect with years of experience. 
            You make final decisions on code quality, select the best solution from multiple 
            options, and provide final recommendations and documentation.""",
            llm=reviewer_llm,
            verbose=True,
            allow_delegation=False
        )
        
        return [research_analyst, claude_generator, gpt_generator, quality_scorer, final_reviewer]
    
    def _create_tasks(self, agents) -> List[Task]:
        """Create generation tasks."""
        research_analyst, claude_generator, gpt_generator, quality_scorer, final_reviewer = agents
        
        # Research and analysis task
        research_task = Task(
            description="""Analyze the user request: {requirements}
            
            Research the DataRobot APIs and methods needed to fulfill this request. Use the retriever 
            system to find relevant documentation, examples, and best practices.
            
            Provide:
            1. Detailed analysis of requirements
            2. List of DataRobot APIs/methods needed  
            3. Key parameters and configurations
            4. Potential challenges or considerations
            5. Recommended approach
            
            Be specific about DataRobot SDK classes, methods, and configuration parameters.""",
            agent=research_analyst,
            expected_output="Detailed technical analysis with specific DataRobot API requirements"
        )
        
        # Claude code generation
        claude_task = Task(
            description="""Based on the research analysis, generate complete DataRobot Python SDK code.
            
            Requirements:
            - Use proper DataRobot SDK patterns and best practices
            - Include proper imports and error handling
            - Add clear documentation and comments
            - Handle edge cases appropriately
            - Follow DataRobot time series framework if applicable
            
            Generate production-ready code that directly addresses the user requirements.""",
            agent=claude_generator,
            expected_output="Complete, production-ready DataRobot Python SDK code with documentation"
        )
        
        # GPT code generation
        gpt_task = Task(
            description="""Based on the research analysis, generate robust DataRobot Python SDK code.
            
            Requirements:
            - Implement comprehensive error handling and logging
            - Create modular, reusable code patterns
            - Include input validation and safety checks
            - Add detailed documentation and type hints
            - Consider enterprise deployment scenarios
            
            Generate enterprise-grade code with focus on robustness and maintainability.""",
            agent=gpt_generator,
            expected_output="Robust, enterprise-grade DataRobot Python SDK code with comprehensive error handling"
        )
        
        # Quality scoring task
        scoring_task = Task(
            description="""Evaluate both code solutions from Claude and GPT generators.
            
            Score each solution (1-10) based on:
            1. Correctness and accuracy
            2. DataRobot best practices adherence
            3. Code quality and readability
            4. Error handling completeness
            5. Documentation quality
            6. Addressing user requirements
            
            Provide detailed scoring with justification for each criterion.""",
            agent=quality_scorer,
            expected_output="Detailed quality scores and analysis for both code solutions"
        )
        
        # Final review task
        final_task = Task(
            description="""Review all code solutions and select the best one.
            
            Consider:
            - Quality scores from the analyst
            - Specific user requirements
            - Code completeness and correctness
            - Best practices adherence
            
            Provide the final recommended solution with:
            1. Selected code solution
            2. Justification for selection
            3. Any final improvements or recommendations
            4. Usage instructions""",
            agent=final_reviewer,
            expected_output="Final selected DataRobot code solution with justification and usage instructions"
        )
        
        return [research_task, claude_task, gpt_task, scoring_task, final_task]
    
    def _search_documentation(self, query: str, top_k: int = 5) -> List[str]:
        """Search documentation using retriever"""
        if not self.retriever:
            return []
        
        results = self.retriever.search(query, top_k=top_k)
        return [f"Title: {r.document.title}\nContent: {r.document.content}" for r in results]
    
    def generate_code(self, user_request: str) -> Dict:
        """
        Generate DataRobot API code from user request.
        
        Args:
            user_request: Natural language request
            
        Returns:
            Dict containing generated code and metadata
        """
        try:
            # Add documentation context if retriever is available
            doc_context = ""
            if self.retriever:
                relevant_docs = self._search_documentation(user_request)
                if relevant_docs:
                    doc_context = f"\n\nRelevant Documentation:\n" + "\n---\n".join(relevant_docs)
            
            # Add documentation context to the request
            enriched_request = user_request + doc_context
            
            # Run the crew
            result = self.crew.kickoff(inputs={"requirements": enriched_request})
            
            return {
                'status': 'success',
                'code': result,
                'models_used': list(self.models.values()),
                'retriever_used': self.retriever is not None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'models_used': list(self.models.values()),
                'retriever_used': self.retriever is not None
            }

# Simple test function
def test_basic_generation():
    """Test basic code generation without retriever"""
    crew = DataRobotAPIGenerationCrew()
    
    test_request = "Create a DataRobot project from a CSV file and start autopilot"
    
    print("🚀 Testing DataRobot API Generation Crew...")
    result = crew.generate_code(test_request)
    
    print(f"\n✅ Result: {result['status']}")
    if result['status'] == 'success':
        print(f"Generated code using models: {result['models_used']}")
    else:
        print(f"Error: {result['error']}")
    
    return result

if __name__ == "__main__":
    test_basic_generation()