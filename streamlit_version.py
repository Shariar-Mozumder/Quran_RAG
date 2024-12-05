import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline


def load_quran_data(excel_file):
    df = pd.read_csv(excel_file)
    df['text'] = df.apply(lambda row: f"Chapter {row['Surah']} ({row['Name']}), Verse {row['Ayat']}: {row['Translation1']}, Explanation: {row['Tafaseer1']}", axis=1)
    return df


def create_embedding_and_index(df):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=df['text'].tolist(), embedding=embedding_model)
    return vector_store


def initialize_llm():
    generator = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-1B-Instruct",
        truncation=True,
        max_length=2048,
        no_repeat_ngram_size=3,
        temperature=0.5,
        top_p=0.8,
        repetition_penalty=1.3,
        num_beams=4
    )
    return HuggingFacePipeline(pipeline=generator)


def create_prompt():
    template = """
    You are a highly knowledgeable Islamic scholar, specializing in providing professional and detailed responses to questions about Islamic rules, regulations, and history. Your answers should strictly adhere to the Quran and authentic Hadith references, providing and mentioning clear explanations and quoting the specific verses of Quran or Hadith references in every response. 

    Do not include unnecessary information or repeat the question or context in your response. Speak concisely, professionally, and with the authority of an expert lecturer. Always back every explanation with a clear and accurate Quranic verse or Hadith reference.

    Query: {query}

    Relevant Verses and Explanations:
    {context}

    Response:
    """
    return PromptTemplate(template=template, input_variables=["query", "context"])


def retrieve_and_generate(vector_store, llm, prompt, query):
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant verse found."
    filled_prompt = prompt.format(query=query, context=context)
    result = llm(filled_prompt)
    response_start = result.find("Response:") + len("Response:")
    response = result[response_start:].strip()
    return response


# Streamlit App
def main():
    st.title("Quran Chatbot")
    st.write("Please enter your question regarding Islam and the Quran.")
    
    # Load the dataset and initialize models
    @st.cache_resource
    def initialize_models():
        quran_df = load_quran_data("main_df.csv")  # Replace with your file path
        vector_store = create_embedding_and_index(quran_df)
        llm = initialize_llm()
        prompt = create_prompt()
        return vector_store, llm, prompt

    vector_store, llm, prompt = initialize_models()

    # Input box for the user's query
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Generating response..."):
            response = retrieve_and_generate(vector_store, llm, prompt, query)
            st.markdown("### Chatbot Response:")
            st.write(response)


if __name__ == "__main__":
    main()
