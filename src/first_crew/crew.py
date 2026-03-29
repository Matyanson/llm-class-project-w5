from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os
from langchain_huggingface import HuggingFaceEmbeddings
from crewai_tools import JSONSearchTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# 1. Bypass the underlying issue in CrewAI-Tools that forcefully checks for an OpenAI Key
os.environ["OPENAI_API_KEY"] = "NA"

# 2. Configure the global Embedding Model (Used for background retrieval such as CrewAI Knowledge)
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)

# 3. Dedicated configuration file for RAG Tools (Dictionary format)
# Since we set MODEL=ollama/phi3 in .env, there is no need to specify the LLM Provider here; the tool will automatically fallback to using Ollama.
rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer", # Specify the use of purely local CPU for automatic chunk generation
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

# 4. [IMPORTANT] Ensure an independent name (.name) and description (.description) is set for each retrieval tool
user_rag_tool = JSONSearchTool(json_path='data/user_subset.json', collection_name='v3_hf_user_data', config=rag_config)
user_rag_tool.name = "search_user_profile_data"
user_rag_tool.description = "Useful to retrieve a specific user's giving habits, average stars, and review counts."

item_rag_tool = JSONSearchTool(json_path='data/item_subset.json', collection_name='v3_hf_item_data', config=rag_config)
item_rag_tool.name = "search_restaurant_feature_data"
item_rag_tool.description = "Useful to retrieve a specific restaurant's location, categories, attributes, and overall stars."

review_rag_tool = JSONSearchTool(json_path='data/review_subset.json', collection_name='v3_hf_review_data', config=rag_config)
review_rag_tool.name = "search_historical_reviews_data"
review_rag_tool.description = "Useful to retrieve the actual text content of past reviews for users or restaurants."

@CrewBase
class CrewProject2():
    """CrewProject2 crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewProject2 crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
