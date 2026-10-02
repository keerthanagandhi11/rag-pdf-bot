from utils.config import GOOGLE_API_KEY, GROQ_API_KEY

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


class RetrievalChainAdapter:
  """Compatibility wrapper for the app's existing chain.invoke({"input": ...}) interface."""

  def __init__(self, llm, retriever, prompt):
    self.llm = llm
    self.retriever = retriever
    self.prompt = prompt

  def invoke(self, payload):
    question = payload["input"]
    docs = self.retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = self.prompt.format_messages(context=context, input=question)
    response = self.llm.invoke(messages)
    answer = getattr(response, "content", str(response))
    if isinstance(answer, list):
      answer = "".join(getattr(item, "text", str(item)) for item in answer)
    return {"answer": answer}


def get_llm_chain(model_provider, model, vectorstore):
  """
  Builds and returns a retrieval chain adapter compatible with the app's current call pattern.

  Parameters:
  - model_provider (str): The provider for the LLM ("groq" or "gemini").
  - model (str): The specific model name.
  - vectorstore (VectorStore): A Chroma vectorstore object for document retrieval.

  Returns:
  - An object exposing an invoke({"input": ...}) method that returns {"answer": ...}.
  """
  prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer as detailed as possible using the context below. If unknown, say 'I don't know.'"),
    ("human", "Context:\n{context}\n\n\nQuestion:\n{input}")
  ])

  if not model:
    return None

  if model_provider == "groq":
    llm = ChatGroq(model=model, api_key=GROQ_API_KEY)
  elif model_provider == "gemini":
    llm = ChatGoogleGenerativeAI(model=model, api_key=GOOGLE_API_KEY)
  else:
    return None

  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
  return RetrievalChainAdapter(llm=llm, retriever=retriever, prompt=prompt)
