import os

from utils.config import MODEL_OPTIONS
from utils.pdf_handler import get_pdf_text, get_text_chunks

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Mapping each model provider to its corresponding persistent directory for storing vectorstore data
PERSIST_DIR = {
  key.lower(): f"./data/{key.lower()}_vector_store.chroma"
  for key in MODEL_OPTIONS.keys()
}

def get_embeddings(model_provider):
  """
  Returns the appropriate embedding model based on the selected model provider.

  - For supported providers, returns a local HuggingFace MiniLM embedding model.

  Raises:
    ValueError: If the given provider is not supported.
  """
  if model_provider not in PERSIST_DIR:
    raise ValueError("Unsupported Model Provider")

  return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")


def get_or_create_vectorstore(uploaded_files, model_provider):
  """
  Loads an existing Chroma vectorstore from disk if it exists, or creates a new one from uploaded PDFs.

  This function:
  - Extracts raw text from uploaded PDFs.
  - Splits the text into chunks suitable for embedding.
  - Loads or creates a vectorstore for the given model provider.
  - Appends to existing vectorstore if already present.

  Args:
    uploaded_files (list): List of uploaded PDF files.
    model_provider (str): Lowercase name of the selected model provider ('groq' or 'gemini').

  Returns:
    Chroma: A Chroma vectorstore containing embedded PDF text chunks.
  """
  # Extract raw text from the uploaded PDF files
  raw_text = get_pdf_text(uploaded_files)

  # Chunk the raw text for embedding (e.g., 5000 characters with overlap)
  chunks = get_text_chunks(raw_text)

  # Load the appropriate embedding model
  embedding = get_embeddings(model_provider)

  # Define directory path to store or retrieve Chroma DB
  persist_path = PERSIST_DIR[model_provider]

  # If the vectorstore directory exists and is not empty, load and append new chunks
  if os.path.exists(persist_path) and os.listdir(persist_path):
    vectorstore = Chroma(
      persist_directory=persist_path,
      embedding_function=embedding
    )
    vectorstore.add_texts(chunks)
  else:
    # Otherwise, create a new vectorstore from the chunks
    vectorstore = Chroma.from_texts(
      texts=chunks,
      embedding=embedding,
      persist_directory=persist_path
    )

  return vectorstore
