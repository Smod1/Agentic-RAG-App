import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai import Agent, Task, Crew
from crewai_tools import PDFSearchTool
from crewai.tools import tool
from dotenv import load_dotenv

# loads API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Set it as an environment variable before running."
    )

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY is not set. "
        "Set it as an environment variable before running."
    )


# creates the llm model, uses groq llama3
print("About to create LLM", flush = True)
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1000,
)
print("LLM created", flush = True)


# PDF RAG tool says to search for information through doc.pdf
rag_tool = PDFSearchTool(
    pdf="doc.pdf",
    config={
        "embedding_model": {
            "provider": "sentence-transformer",
            "config": {
                "model_name": "BAAI/bge-small-en-v1.5",
            },
        },
        "vectordb": {
            "provider": "chromadb",
            "config": {},
        },
    },
)
print("PDFSearchTool Created")

# creates the tool to enable web search, returns 3 search results
web_search_tool = TavilySearchResults(
    k=3
)


# creates router tool
@tool
def router_tool(question: str) -> str:
    # Decide whether a question should use the PDF vectorstore
    # or an external web search.


    question_lower = question.lower()

    # Questions specifically about the contents of the
    # provided Sporo Health PDF should use the vectorstore.
    pdf_keywords = [
        "sporo",
        "patient chart",
        "chart review",
        "clinical summarization",
        "clinical summary",
        "medical management",
        "diagnostic accuracy",
        "physician burnout",
        "ehr",
        "electronic health record",
        "clinical workflow",
        "medical errors",
        "ai clinical",
    ]

    for keyword in pdf_keywords:
        if keyword in question_lower:
            return "vectorstore"

    return "websearch"


# creates agent which decides whether to web search or use document
Router_Agent = Agent(
    role="Question Router",

    goal=(
        "Determine whether the user's question should be answered "
        "using the provided PDF vectorstore or an external web search."
    ),

    backstory=(
        "You are an expert information-routing agent. "
        "The PDF contains research about clinical summarization, "
        "patient chart review, healthcare information overload, "
        "AI-assisted clinical workflows, and Sporo Health. "
        "Questions about those topics should use the PDF vectorstore. "
        "Questions requiring information outside the PDF should use web search."
    ),

    verbose=True,
    allow_delegation=False,
    llm=llm,

    tools=[router_tool],
)


# retriever agent
Retriever_Agent = Agent(
    role="Information Retriever",

    goal=(
        "Retrieve accurate information needed to answer the user's question."
    ),

    backstory=(
        "You are an expert research assistant. "
        "You have access to a PDF search tool and a web search tool. "
        "When the router selects vectorstore, search the PDF carefully. "
        "When the router selects websearch, use Tavily to retrieve current "
        "information from the internet. "
        "Do not invent facts that are not supported by the retrieved information."
    ),

    verbose=True,
    allow_delegation=False,
    llm=llm,

    # IMPORTANT:
    # The agent MUST have the tools.
    tools=[rag_tool, web_search_tool],
)


# creates the agent which determines how relevant the information retrieved is to the question
# to ensure the answer is left with no gaps

Grader_agent = Agent(
    role="Retrieval Relevance Grader",

    goal=(
        "Determine whether the retrieved information is relevant "
        "to the user's question."
    ),

    backstory=(
        "You are a strict but fair evaluator of retrieved information. "
        "Check whether the retrieved information actually addresses "
        "the user's question. Do not judge based merely on matching "
        "one keyword."
    ),

    verbose=True,
    allow_delegation=False,
    llm=llm,
)


# creates the agent which determines whether or not the answer is hallucinated
hallucination_grader = Agent(
    role="Hallucination Grader",

    goal=(
        "Determine whether the retrieved answer is grounded in "
        "the information returned by the retrieval step."
    ),

    backstory=(
        "You carefully compare the proposed answer with the retrieved "
        "facts. An answer is grounded only when its claims are supported "
        "by the retrieved information. Do not accept unsupported claims."
    ),

    verbose=True,
    allow_delegation=False,
    llm=llm,
)


# generates the final answer using the retrieved information
answer_grader = Agent(
    role="Final Answer Generator",

    goal=(
        "Generate a clear, accurate, concise answer to the user's question."
    ),

    backstory=(
        "You are the final answer specialist. "
        "Use validated retrieved information to answer the user. "
        "If the retrieved information is insufficient, use web search "
        "to obtain additional information. "
        "Never invent facts."
    ),

    verbose=True,
    allow_delegation=False,
    llm=llm,

    # The final agent needs Tavily because it may need
    # to perform a fallback web search.
    tools=[web_search_tool],
)


# sets the task to determine which router is optimal, includes a description of which router should be used for
# which question

router_task = Task(
    description=(
        "Analyze the following user question:\n\n"
        "{question}\n\n"

        "Determine which information source should be used.\n\n"

        "Use 'vectorstore' when the question can be answered from "
        "the provided PDF about Sporo Health, patient chart review, "
        "clinical summarization, EHRs, AI-assisted clinical workflow, "
        "medical errors, physician burnout, or related topics.\n\n"

        "Use 'websearch' when the information is not contained in "
        "the provided PDF or requires current external information.\n\n"

        "Return exactly one word:\n"
        "vectorstore\n"
        "or\n"
        "websearch"
    ),

    expected_output=(
        "Exactly one word: vectorstore or websearch."
    ),

    agent=Router_Agent,

    tools=[router_tool],
)


# sets the task to retrieve information using the proper tools, to ensure no hallucinations or content is assumed
retriever_task = Task(
    description=(
        "Answer the user's question using the appropriate retrieval tool.\n\n"

        "User question:\n"
        "{question}\n\n"

        "First examine the routing decision from router_task.\n\n"

        "If the routing decision is 'vectorstore', use the PDFSearchTool "
        "to search doc.pdf and retrieve the most relevant information.\n\n"

        "If the routing decision is 'websearch', use the Tavily web "
        "search tool to retrieve relevant information from the internet.\n\n"

        "Do not make up information.\n"
        "Return the relevant retrieved facts and enough context for "
        "another agent to evaluate them."
    ),

    expected_output=(
        "Relevant factual information retrieved from either the PDF "
        "vectorstore or web search."
    ),

    agent=Retriever_Agent,

    context=[router_task],
)


# sets task to say whether the information retrieved is relevant or not
grader_task = Task(
    description=(
        "Evaluate the retrieved information from retriever_task "
        "against the user's question.\n\n"

        "User question:\n"
        "{question}\n\n"

        "Determine whether the retrieved information is relevant "
        "and useful for answering the question.\n\n"

        "Return 'yes' if the retrieved information is relevant.\n"
        "Return 'no' if it is not relevant.\n\n"

        "Return ONLY yes or no."
    ),

    expected_output=(
        "Exactly one word: yes or no."
    ),

    agent=Grader_agent,

    context=[retriever_task],
)


# sets the task to check for hallucination or actually an answer
hallucination_task = Task(
    description=(
        "Check whether the information retrieved by retriever_task "
        "is sufficiently grounded to support an answer to the user's "
        "question.\n\n"

        "User question:\n"
        "{question}\n\n"

        "Review the actual retrieved information from retriever_task. "
        "Do not rely only on the relevance grader's yes/no result.\n\n"

        "Return 'yes' if the retrieved information supports a factual "
        "answer to the question.\n\n"

        "Return 'no' if the retrieved information is insufficient, "
        "unsupported, or unrelated.\n\n"

        "Return ONLY yes or no."
    ),

    expected_output=(
        "Exactly one word: yes or no."
    ),

    agent=hallucination_grader,

    context=[
        retriever_task,
        grader_task,
    ],
)


# sets the task to produce the final answer

answer_task = Task(
    description=(
        "Produce the final answer to the user's question.\n\n"

        "User question:\n"
        "{question}\n\n"

        "Use the retrieved information and the grading results.\n\n"

        "If the retrieved information is relevant and grounded, "
        "answer the question clearly and concisely using that information.\n\n"

        "If the retrieved information is insufficient or not grounded, "
        "use the web search tool to find additional information and "
        "then answer the question.\n\n"

        "Do not mention internal agents, routing, graders, vectorstores, "
        "or this workflow unless the user specifically asks about them.\n\n"

        "Do not invent facts."
    ),

    expected_output=(
        "A clear, concise, factual answer to the user's question."
    ),

    agent=answer_grader,

    context=[
        retriever_task,
        grader_task,
        hallucination_task,
    ],
)


# create the crew to work together, each agent with different responsibilities

rag_crew = Crew(
    agents=[
        Router_Agent,
        Retriever_Agent,
        Grader_agent,
        hallucination_grader,
        answer_grader,
    ],

    tasks=[
        router_task,
        retriever_task,
        grader_task,
        hallucination_task,
        answer_task,
    ],

    verbose=True,
)

# test an input query
inputs = {
    "question": "Does Sporo streamline patient chart reviews?"
}
result = rag_crew.kickoff(
    inputs=inputs
)



# output the result
print("\n")
print("FINAL ANSWER")
print("\n")
print(result)
