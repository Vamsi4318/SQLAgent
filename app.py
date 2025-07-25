import pandas as pd
import sqlite3
import io
import base64
import matplotlib.pyplot as plt
import streamlit as st

from langchain.agents import tool, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = ""

def load_file_to_sqlite(file_obj):
    filename = file_obj.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(file_obj)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file_obj, engine="openpyxl")
    elif filename.endswith(".xls"):
        df = pd.read_excel(file_obj, engine="xlrd")
    else:
        raise ValueError("Unsupported file format. Please upload a CSV, XLSX, or XLS.")
    conn = sqlite3.connect(":memory:")
    df.to_sql("data", conn, index=False, if_exists='replace')
    return conn, df


def make_tools(conn):
    @tool
    def run_sql_query(query: str) -> str:
        """Run a SQL query on the uploaded file. Input: SQL SELECT query."""
        try:
            result = pd.read_sql_query(query, conn)
            return result.to_string(index=False)
        except Exception as e:
            return f"SQL Error: {e}"

    @tool
    def generate_chart(instruction: str) -> str:
        """
        Generate a chart from the data.
        Input format: 'chart_type=bar; query=SELECT col1, col2 FROM data GROUP BY col1'
        Supported types: bar, line, pie
        """
        try:
            parts = instruction.split(";")
            chart_type = ""
            query = ""
            for part in parts:
                if "chart_type=" in part:
                    chart_type = part.split("=", 1)[1].strip().lower()
                if "query=" in part:
                    query = part.split("=", 1)[1].strip()
            if not chart_type or not query:
                return "Instruction must include chart_type and query."

            df = pd.read_sql_query(query, conn)
            if df.empty or df.shape[1] < 2:
                return "Not enough data to plot."

            # Plot
            plt.clf()
            if chart_type == "bar":
                df.plot(x=df.columns[0], y=df.columns[1], kind="bar", legend=False)
            elif chart_type == "line":
                df.plot(x=df.columns[0], y=df.columns[1], kind="line", legend=False)
            elif chart_type == "pie":
                df.set_index(df.columns[0]).plot.pie(y=df.columns[1], autopct='%1.1f%%')
            else:
                return "Unsupported chart type."

            # Convert to base64
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            return f"Chart Error: {e}"

    return [run_sql_query, generate_chart]


def run_agent_on_question(file_obj, user_question: str):
    conn, df = load_file_to_sqlite(file_obj)
    tools = make_tools(conn)

    agent = initialize_agent(
        tools=tools,
        llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro"),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    tools_dict = {t.name: t for t in tools}

    chart_prompt = f"""
        You are an assistant that extracts chart generation instructions.
        Return format: chart_type=<bar|line|pie>; query=<SQL query over table 'data'>
        Example: chart_type=bar; query=SELECT region, SUM(sales) FROM data GROUP BY region

        Now convert this user request into that format:
        "{user_question}"
        """
    try:
        response = agent.run(chart_prompt)
        if "chart_type=" in response and "query=" in response:
            return tools_dict["generate_chart"].run(response)
        else:
            return agent.run(user_question)  # Fallback
    except Exception as e:
        return f"Agent Error: {e}"

st.set_page_config(page_title="Gemini Data Agent", layout="centered")
st.title("📊 Gemini-Powered Data Assistant")

# Session state
if "result" not in st.session_state:
    st.session_state.result = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "question" not in st.session_state:
    st.session_state.question = ""

# File Upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    try:
        df_preview = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        with st.expander("🔍 Preview Uploaded Data", expanded=False):
            st.dataframe(df_preview.head(10))
    except Exception as e:
        st.error(f"Failed to preview file: {e}")

# Question input
st.session_state.question = st.text_input("Ask a question or request a chart", value=st.session_state.question)

# Submit & Reset buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🚀 Submit"):
        if st.session_state.uploaded_file and st.session_state.question.strip():
            with st.spinner("Thinking..."):
                output = run_agent_on_question(st.session_state.uploaded_file, st.session_state.question)
                st.session_state.result = output
        else:
            st.warning("Please upload a file and enter a question.")

with col2:
    if st.button("🔄 Reset"):
        st.session_state.result = None
        st.session_state.uploaded_file = None
        st.session_state.question = ""
        st.rerun()

# Output
if st.session_state.result:
    if isinstance(st.session_state.result, str) and st.session_state.result.startswith("data:image/png"):
        st.image(st.session_state.result)
    else:
        st.text_area("💡 Output", st.session_state.result, height=300)