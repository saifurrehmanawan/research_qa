import streamlit as st
import ast
import pandas as pd
import feedparser
import arxiv
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.runnables import ConfigurableField
#import streamlit as st
import faiss
import pandas
import re
from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.utils import DistanceStrategy
import google.generativeai as genai
from IPython.display import display, Markdown
import urllib # Import urllib to use url encoding function
import os
import pandas as pd
import csv
# Import the Python SDK
import google.generativeai as genai
# Used to securely store your API key
import os
# Load the API key from Streamlit secrets
api_key = st.secrets["API_KEY"]
genai.configure(api_key=api_key)
class research_paper_qa:
  def __init__(self):
    self.model = genai.GenerativeModel('gemini-1.5-flash')
    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  def get_search_queries(self, question):
    query = f'''
        Based on the question "{question}", please provide a concise list of relevant search queries that can be used to find related papers on arXiv.
        You must make the Python list without any additional text or explanation.
        The response must not be in the markdown format.
        please do not start with "```python".
      '''

    response = self.model.generate_content(query)
    return ast.literal_eval(response.text)


    # Convert the response text to a Python list
    response_list = ast.literal_eval(response.text)
    # Ensure elements in the list are properly quoted
    response_list = [element.replace("'", "\\'") for element in response_list]
    return response_list

  def fetch_arxiv_papers(self, keywords):
    # Define the arXiv API query URL
    base_url = 'http://export.arxiv.org/api/query?'
    titles = []
    abstracts = []
    for keyword in keywords:
      # Use urllib.parse.quote to encode the keyword
      query = f'search_query=all:{urllib.parse.quote(keyword)}&start=0&max_results=10'
      url = base_url + query
      # Fetch and parse the RSS feed
      feed = feedparser.parse(url)
      # Extract relevant information from the feed
      for entry in feed.entries:
        titles.append(entry.title)
        abstracts.append(entry.summary)
      # Create a DataFrame from the lists
      df = pd.DataFrame({
        'Title': titles,
        'Abstract': abstracts
      })
    return df
  def ret_docs(self, df):
    loader = DataFrameLoader(df, page_content_column="Title")
    documents = loader.load()
    num_docs = 5 # Default number of documents to retrieve
    bm25_retriever = BM25Retriever.from_documents(
      documents
      ).configurable_fields(
      k=ConfigurableField(
        id="search_kwargs_bm25",
        name="k",
        description="The search kwargs to use",
      )
    )
    faiss_vectorstore = FAISS.from_documents(
      documents, self.embeddings, distance_strategy=DistanceStrategy.COSINE
    )
    faiss_retriever = faiss_vectorstore.as_retriever(
      search_kwargs={"k": num_docs}
      ).configurable_fields(
      search_kwargs=ConfigurableField(
        id="search_kwargs_faiss",
        name="Search Kwargs",
        description="The search kwargs to use",
      )
    )
    # initialize the ensemble retriever
    vector_database = EnsembleRetriever(
      retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5] # You can adjust the weight of each retriever in the EnsembleRetriever
    )
    return vector_database
  def retrieve(self, query, vector_database):
    # Retrieve top k similar documents to query
    #docs = retriever.get_relevant_documents(query)
    config = {"configurable": {"search_kwargs_faiss": {"k": 5}, "search_kwargs_bm25": 5}}
    retrieved_docs = vector_database.invoke(query, config=config)
    return retrieved_docs[0]
  def title_extract(self, doc):
    # Extract the page content
    start = doc.find("page_content='") + len("page_content='")
    end = doc.find("' metadata=")
    title = doc[start:end]
    
    # Remove newline characters from the title
    title = title.replace('\n', '')
    
    return title
  def download_arxiv_paper(self, title):
    # Search for the paper by title
    search = arxiv.Search(
        query=title,
        max_results=1,
        #sort_by=arxiv.SortCriterion.SubmittedDate
    )
    # Fetch the first result (if any)
    for result in search.results():
        print(f"Found paper: {result.title}")
        print(f"Downloading from: {result.pdf_url}")
        # Download the PDF
        filename=f"{result.title.replace(' ', '_')}.pdf"
        result.download_pdf(filename=f"{result.title.replace(' ', '_')}.pdf")
        print(f"Downloaded: {result.title}.pdf")
        return filename
    
    print("No paper found with that title.")
  def rp_qa(self, question, filename, title):
    # Upload the file and print a confirmation
    sample_file = genai.upload_file(path=filename,
                                display_name=title)
    print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
    # Prompt the model with text and the previously uploaded image.
    prompt = f"""
              Answer the question: {question} using general knowledge and established scientific principles.

              Present the information clearly and in a manner that is easy to understand. Avoid mentioning any specific papers, theories, or technical terms that might be unfamiliar to the user. Instead, provide a comprehensive and straightforward explanation, including necessary details such as key concepts, methods, and equations, if applicable.

              Ensure the response is formatted in markdown and is suitable for users with a basic understanding of the topic.

             """

    response = self.model.generate_content([sample_file, prompt])
    # Display the Markdown content
    st.markdown(response.text)
  def del_file(self, filename):
    os.remove(filename)
