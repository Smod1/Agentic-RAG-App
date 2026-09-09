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
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1000,
)

# PDF RAG
rag_tool = PDFSearchTool(
    pdf="doc.pdf"
)

# Agent
agent = Agent(
    role="PDF Researcher",
    goal="Answer questions using the provided PDF",
    backstory="You are an expert document researcher.",
    tools=[rag_tool],
    llm=llm,
)