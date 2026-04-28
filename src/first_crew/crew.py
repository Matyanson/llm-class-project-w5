from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from crewai_tools import RagTool
from crewai_tools.tools.rag import ProviderSpec, RagToolConfig, VectorDbConfig
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai_tools import JSONSearchTool

load_dotenv()

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# === LLM Provider Selection ===
llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

if llm_provider == "nvidia":
    # Route through LiteLLM's OpenAI-compatible interface to Nvidia API
    os.environ["MODEL"] = f"openai/{os.getenv('NVIDIA_MODEL_NAME', 'meta/llama-3.1-8b-instruct')}"
    os.environ["OPENAI_API_BASE"] = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
    os.environ["OPENAI_API_KEY"] = os.getenv("NVIDIA_API_KEY", "")
else:
    # Default to local Ollama Phi3
    os.environ["MODEL"] = os.getenv('OLLAMA_MODEL', "ollama/phi3")

# Workaround for early CrewAI-Tools versions that enforce OpenAI Key validation via Pydantic
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "NA")

# 2. Configure the global Embedding Model (Used for background retrieval such as CrewAI Knowledge)
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)

# 3. Load the translation guide as global knowledge
yelp_translation_knowledge = TextFileKnowledgeSource(
    file_paths=["yelp_data_translation.md"]
)


# 3. Dedicated configuration file for RAG Tools (Dictionary format)

rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

def create_rag_tool(json_path: str, collection_name: str, config: dict, name: str, description: str) -> JSONSearchTool:
    from crewai.utilities.paths import db_storage_path
    from crewai_tools.tools.json_search_tool.json_search_tool import FixedJSONSearchToolSchema
    import sqlite3
    import os
    
    collection_exists = False
    db_file = os.path.join(db_storage_path(), "chroma.sqlite3")
    
    if os.path.exists(db_file):
        try:
            # Check native sqlite3 for existing collection to heavily avoid 100% JSON text synchronous chunking bottleneck
            # and avoid ChromaDB singleton initialization conflicts with CrewAI's internal Settings
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
            if cursor.fetchone() is not None:
                collection_exists = True
            conn.close()
        except Exception:
            pass

    if collection_exists:
        tool = JSONSearchTool(collection_name=collection_name, config=config)
        # CRITICAL: Force the Pydantic schema to hide json_path from the Agent, 
        # so it doesn't trigger validation errors or pass the path and trigger the 3-hour hash loop!
        tool.args_schema = FixedJSONSearchToolSchema
    else:
        tool = JSONSearchTool(json_path=json_path, collection_name=collection_name, config=config)
        
    tool.name = name
    tool.description = description
    return tool



print("setting up retrieval tools...")
# 4. [IMPORTANT] Ensure an independent name (.name) and description (.description) is set for each retrieval tool

# User profile tool
user_rag_tool = create_rag_tool(
    json_path='data/filtered_user.json',
    collection_name='benchmark_true_fresh_index_Filtered_User_1',
    config=rag_config,
    name="search_user_profile_data",
    description=(
        "Searches the user profile database using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'What are the review habits and average stars for user _BcWyKQL16?'. "
        "Do NOT pass raw user_id or JSON objects directly."
    )
)

# Restaurant feature tool
item_rag_tool = create_rag_tool(
    json_path='data/filtered_item.json',
    collection_name='benchmark_true_fresh_index_Filtered_Item_1',
    config=rag_config,
    name="search_restaurant_feature_data",
    description=(
        "Searches the restaurant/business database using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'What are the categories, location, and star rating for business abc123?'. "
        "Do NOT pass raw item_id or JSON objects directly."
    )
)

# Reviews tool
review_rag_tool = create_rag_tool(
    json_path='data/test_review.json',
    collection_name='benchmark_true_fresh_index_Filtered_Review_1',
    config=rag_config,
    name="search_historical_reviews_data",
    description=(
        "Searches historical review texts using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'Find past reviews written by user _BcWyKQL16 about food quality and service'. "
        "Do NOT pass raw user_id, item_id, or JSON objects directly."
    )
)

print("retrieval tools setup complete!")

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
            config=self.agents_config['researcher'], # type: ignore[index],
            tools=[user_rag_tool, item_rag_tool, review_rag_tool],
            verbose=True,
            allow_delegation=False
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index],
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
            knowledge_sources=[yelp_translation_knowledge],
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
