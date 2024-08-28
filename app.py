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

    st.title("Arxiv Q&A Assistant")
    st.info("The Arxiv Q&A Assistant is an AI-driven tool designed to answer user questions by searching and extracting relevant information from research papers on arXiv. By leveraging advanced natural language processing techniques, this program allows users to interact with academic research more effectively, providing concise and accurate answers to their queries based on the latest scientific literature. Ideal for researchers, students, and academics, it simplifies the process of finding precise information within a vast array of research documents.")

    # User input for the question
    question = st.text_input("Ask a question about astronomy:")

    if st.button("Get Answer"):
        # Styled message for extracting keywords
        st.markdown(
            '<p style="font-family:Courier; color:blue; font-size:20px;">Extracting keywords...</p>',
            unsafe_allow_html=True
        )
        keywords_list = research_paper_qa.get_keywords(question)
    
        # Styled message for fetching arxiv papers
        st.markdown(
            '<p style="font-family:Courier; color:blue; font-size:20px;">Fetching arxiv papers...</p>',
            unsafe_allow_html=True
        )
        df = research_paper_qa.fetch_arxiv_papers(keywords_list)
    
        # Styled message for creating vector database
        st.markdown(
            '<p style="font-family:Courier; color:blue; font-size:20px;">Creating vector database...</p>',
            unsafe_allow_html=True
        )
        vector_database = research_paper_qa.ret_docs(df)
    
        # Styled message for retrieving relevant arxiv paper
        st.markdown(
            '<p style="font-family:Courier; color:blue; font-size:20px;">Retrieving relevant arxiv paper...</p>',
            unsafe_allow_html=True
        )
        doc = research_paper_qa.retrieve(question, vector_database)
        titles = []
        for paper in doc:
          #print("paper: ", paper)
          title = research_paper_qa.title_extract(str(doc))
          print(title)
          titles.append(title)
    
        # Styled message for downloading
        st.markdown(
            '<p style="font-family:Courier; color:blue; font-size:20px;">Downloading...</p>',
            unsafe_allow_html=True
        )
        filenames = []
        for title in titles:
          filename = research_paper_qa.download_arxiv_paper(title)
          filenames.append(filename)
    
        # Styled message for answering
        st.markdown(
            '<p style="font-family:Courier; color:blue; font-size:20px;">Answering...</p>',
            unsafe_allow_html=True
        )

        # Display the answer
        research_paper_qa.rp_qa(question, filenames, title)
    
        # Clean up
        research_paper_qa.del_file(filenames)

if __name__ == "__main__":
    main()
