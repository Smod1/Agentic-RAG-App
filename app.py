from langchain_openai import ChatOpenAI
import os
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai import Crew
from crewai import Task
from crewai import Agent
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Set up LLM model using Groq's Llama3  model
llm = ChatOpenAI(
    openai_api_base = "https://api.groq.com/openai/v1",
    openai_api_key = GROQ_API_KEY,
    model_name = "llama3-8b-8192",
    temperature = 0.1,
    max_tokens = 1000,
)

