import streamlit as st
from agents import research_paper_qa

@st.cache(allow_output_mutation=True)
def load_agent():
    return research_paper_qa()

# Initialize the AI agent
research_paper_qa = load_agent()

def main():
    # Add logo at the top
    #st.image("logo.png", width=100)

    st.title("NCBC ASTRO LLAMA")
    st.info("NCBC ASTRO LLAMA is an advanced chatbot designed to answer a wide range of astronomical questions efficiently based on YouTube podcasts, utilizing information from YouTube podcasts.")

    # User input for the question
    user_question = st.text_input("Ask a question about astronomy:")

    if st.button("Get Answer"):
      keywords_list = qa.get_keywords(question)
      df = qa.fetch_arxiv_papers(keywords_list)
      vector_database = qa.ret_docs(df)
      doc = qa.retrieve(question, vector_database)
      title = qa.title_extract(str(doc))
      filename = qa.download_arxiv_paper(title)
      qa.rp_qa(question, filename, title)
      del_file(filename)

if __name__ == "__main__":
    main()
