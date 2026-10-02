"""
Chat Section Module
--------------------

This module handles all logic related to chat functionalities, including:
- Initializing session state
- Rendering chat history
- Handling user queries and responses
- Displaying uploaded PDF files
- Downloading chat history

All UI components are built with Streamlit widgets.
"""

import pandas as pd
import streamlit as st

from datetime import datetime


def setup_session_state():
  """
  Initialize necessary Streamlit session state variables if they are not already defined.
  Ensures stable app behavior across reruns.
  """
  for key, default in {
    "chat_history": [],             # Stores tuples of (question, answer, provider, model, pdfs, timestamp)
    "vector_store": None,           # Stores the vector store instance for PDF embeddings
    "pdf_files": [],                # Currently submitted PDF files
    "last_provider": None,          # Tracks last selected provider for dynamic reloading
    "unsubmitted_files": False,     # Tracks whether new files were uploaded but not submitted
    "uploader_key": 0               # Used to reset file_uploader widget
  }.items():
    if key not in st.session_state:
      st.session_state[key] = default

def render_chat_history():
  """
  Display all previous user and AI messages from chat history.
  """
  for q, a, *_ in st.session_state.get("chat_history", []):
    with st.chat_message("user"):
      st.markdown(q)
    with st.chat_message("ai"):
      st.markdown(a)

def handle_user_input(model_provider, model, chain, fallback_chain=None, fallback_model=None):
  """
  Handles user input from the chat input box. 
  Invokes the LLM chain using the provided question and displays the result.

  Parameters:
  - model_provider (str): The selected LLM provider
  - model (str): The specific model used for answering
  - chain (RetrievalChain): The LangChain retrieval chain for querying vectorstore
  """
  # Disable question input if unsubmitted files or no files uploaded
  disable_question_input = (
    st.session_state.get("unsubmitted_files", False) or
    not st.session_state.get(f"uploaded_files_{st.session_state.uploader_key}", []) or
    not chain
  )

  question = st.chat_input(
    "💬 Ask a Question from the PDF Files",
    disabled=disable_question_input
  )

  if not question:
    return

  with st.chat_message("user"):
    st.markdown(question)
  with st.chat_message("ai"):
    with st.spinner("Thinking..."):
      active_provider = model_provider
      active_model = model
      try:
        output = chain.invoke({"input": question})["answer"]
      except Exception as e:
        error_message = str(e).lower()
        quota_exceeded = "429" in error_message and "quota" in error_message
        if model_provider != "gemini" or fallback_chain is None or not quota_exceeded:
          st.error(f"Error: {str(e)}")
          return

        st.warning(f"Gemini quota reached. Retrying with Groq ({fallback_model}).")
        try:
          output = fallback_chain.invoke({"input": question})["answer"]
          active_provider = "groq"
          active_model = fallback_model
        except Exception as fallback_error:
          st.error(f"Groq fallback failed: {str(fallback_error)}")
          return

      st.markdown(output)
      pdf_names = [f.name for f in st.session_state.get("pdf_files")]
      st.session_state.chat_history.append((question, output, active_provider, active_model, pdf_names, datetime.now()))

def render_uploaded_files_expander():
  """
  Displays the list of successfully uploaded PDF files in an expander.
  Shown only if files are submitted and not pending.
  """
  uploaded_files = st.session_state.get(f"uploaded_files_{st.session_state.uploader_key}", [])
  if uploaded_files and not st.session_state.get("unsubmitted_files"):
    with st.expander("📎 Uploaded Files:"):
      for f in uploaded_files:
        st.markdown(f"- {f.name}")

def render_download_chat_history():
  """
  Adds a button to download chat history as a CSV file with columns: 
  Question, Answer, Model, Model Name, PDF File(s), and Timestamp.
  """
  df = pd.DataFrame(
    st.session_state.chat_history,
    columns=["Question", "Answer", "Model", "Model Name", "PDF File", "Timestamp"]
  )

  with st.expander("📎 Download Chat History:"):
    st.sidebar.download_button(
      "📥 Download Chat History",
      data=df.to_csv(index=False),
      file_name="chat_history.csv",
      mime="text/csv"
    )
