import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from crewai_tools import SerpApiGoogleSearchTool

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Customer Support System",
    page_icon="💬",
    layout="wide"
)


# ============================================================
# 2. Load Environment Variables
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file.")
    st.stop()

if not SERPAPI_API_KEY:
    st.error("SERPAPI_API_KEY not found in .env file.")
    st.stop()


# ============================================================
# 3. Load FAISS Vector Database
# ============================================================

VECTOR_STORE_PATH = "vector_store"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


# ============================================================
# 4. RAG Tool
# ============================================================

class CustomerSupportRAGTool(BaseTool):

    name: str = "Customer Support Knowledge Base"

    description: str = (
        "Search the local customer support knowledge base "
        "using the FAISS vector database. "
        "Use this tool to retrieve information about CASA, "
        "KYC, MFA, password reset, transaction issues and "
        "customer data security."
    )

    def _run(self, query: str) -> str:

        documents = vector_store.similarity_search(
            query,
            k=3
        )

        if not documents:
            return "No relevant information found."

        results = []

        for document in documents:
            results.append(document.page_content)

        return "\n\n---\n\n".join(results)


rag_tool = CustomerSupportRAGTool()


# ============================================================
# 5. Web Search Tool
# ============================================================

web_search_tool = SerpApiGoogleSearchTool()


# ============================================================
# 6. Save File Tool
# ============================================================

class SaveSupportRecordTool(BaseTool):

    name: str = "Save Customer Support Record"

    description: str = (
        "Save the customer query, RAG answer and web search "
        "answer into a text file inside the outputs folder."
    )

    def _run(self, content: str) -> str:

        output_folder = "outputs"

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        file_path = os.path.join(
            output_folder,
            f"customer_support_{timestamp}.txt"
        )

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(content)

        return f"Record saved successfully: {file_path}"


save_tool = SaveSupportRecordTool()


# ============================================================
# 7. Agent 1 - RAG Agent
# ============================================================

rag_agent = Agent(

    role="Customer Support RAG Specialist",

    goal=(
        "Answer customer support questions using "
        "the local customer support knowledge base."
    ),

    backstory=(
        "You are a customer support specialist. "
        "Use the knowledge base tool before answering. "
        "Give clear and accurate answers."
    ),

    tools=[rag_tool],

    llm="gpt-4o-mini",

    verbose=False
)


# ============================================================
# 8. Agent 2 - Web Search Agent
# ============================================================

web_agent = Agent(

    role="Web Research Customer Support Specialist",

    goal=(
        "Search the web for relevant information related "
        "to the customer's question."
    ),

    backstory=(
        "You are a web research specialist. "
        "Use the web search tool to find useful information "
        "and summarize it clearly."
    ),

    tools=[web_search_tool],

    llm="gpt-4o-mini",

    verbose=False
)


# ============================================================
# 9. Agent 3 - Save Agent
# ============================================================

save_agent = Agent(

    role="Customer Support Record Manager",

    goal=(
        "Save the customer query and the answers from the "
        "RAG and web research agents into a text file."
    ),

    backstory=(
        "You maintain customer support records. "
        "Use the file-saving tool to save the complete "
        "customer support interaction."
    ),

    tools=[save_tool],

    llm="gpt-4o-mini",

    verbose=False
)


# ============================================================
# 10. Task 1 - RAG
# ============================================================

rag_task = Task(

    description=(
        "Answer the following customer question:\n\n"
        "{customer_query}\n\n"
        "Use the Customer Support Knowledge Base tool "
        "to retrieve relevant information first. "
        "Then provide a concise customer-friendly answer."
    ),

    expected_output=(
        "A clear answer based on the local knowledge base."
    ),

    agent=rag_agent
)


# ============================================================
# 11. Task 2 - Web Search
# ============================================================

web_task = Task(

    description=(
        "Research the following customer question on the web:\n\n"
        "{customer_query}\n\n"
        "Use the web search tool to find relevant information "
        "and provide a concise summary."
    ),

    expected_output=(
        "A concise answer based on web search results."
    ),

    agent=web_agent,

    context=[rag_task]
)


# ============================================================
# 12. Task 3 - Save Record
# ============================================================

save_task = Task(

    description=(
        "Create a customer support record containing:\n\n"

        "CUSTOMER QUERY:\n"
        "{customer_query}\n\n"

        "RAG ANSWER:\n"
        "Include the answer produced by the RAG agent.\n\n"

        "WEB SEARCH ANSWER:\n"
        "Include the answer produced by the Web Research agent.\n\n"

        "Save the complete record using the "
        "Save Customer Support Record tool."
    ),

    expected_output=(
        "Confirmation that the customer support record "
        "was saved successfully."
    ),

    agent=save_agent,

    context=[
        rag_task,
        web_task
    ]
)


# ============================================================
# 13. Create Sequential Crew
# ============================================================

crew = Crew(

    agents=[
        rag_agent,
        web_agent,
        save_agent
    ],

    tasks=[
        rag_task,
        web_task,
        save_task
    ],

    process="sequential",

    verbose=False
)


# ============================================================
# 14. Streamlit User Interface
# ============================================================

st.title("💬 Customer Support System")

st.write(
    "Ask a customer support question and get answers "
    "from both the internal knowledge base and the web."
)

st.divider()


customer_query = st.text_input(
    "Enter your customer support question:",
    placeholder="Example: What is CASA in banking?"
)


if st.button("🔍 Submit Question", type="primary"):

    if not customer_query.strip():

        st.warning(
            "Please enter a customer support question."
        )

    else:

        with st.spinner(
            "CrewAI agents are processing your question..."
        ):

            result = crew.kickoff(
                inputs={
                    "customer_query": customer_query
                }
            )


        # ----------------------------------------------------
        # Extract individual task outputs
        # ----------------------------------------------------

        rag_answer = result.tasks_output[0].raw
        web_answer = result.tasks_output[1].raw


        # ----------------------------------------------------
        # Display RAG Answer
        # ----------------------------------------------------

        st.subheader("📚 RAG Answer")

        st.info(rag_answer)


        # ----------------------------------------------------
        # Display Web Answer
        # ----------------------------------------------------

        st.subheader("🌐 Web Search Answer")

        st.success(web_answer)


        # ----------------------------------------------------
        # Show save confirmation
        # ----------------------------------------------------

        st.success(
            "✅ Customer support record saved successfully "
            "in the outputs folder."
        )