import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_quran_data(excel_file):
    df = pd.read_csv(excel_file)
    df['text'] = df.apply(lambda row: f"Chapter {row['Surah']} ({row['Name']}), Verse {row['Ayat']}: {row['Translation1']}, Explanation: {row['Tafaseer1']}", axis=1)
    return df


def create_embedding_and_index(df):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=df['text'].tolist(), embedding=embedding_model)
    return vector_store


def initialize_llm():
    # model_name = "mistralai/Mistral-7B-v0.1"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    # generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=True)
    # generator= pipeline("text-generation", model="HuggingFaceTB/SmolLM-360M-Instruct")
    # generator=pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct",truncation=True, max_length=2048)
    # Configure generation pipeline
    generator = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-1B-Instruct",
        truncation=True,
        max_length=2048,  # Reduced to focus on concise responses
        no_repeat_ngram_size=3,  # Avoid repetitive phrasing
        temperature=0.5,  # Lowered for more deterministic and focused outputs
        top_p=0.8,  # Encourages more logical answers
        repetition_penalty=1.3,  # Discourages repetitive patterns
        num_beams=4  # Enables better quality generation with beam search
    )
    
    return HuggingFacePipeline(pipeline=generator)
    # return generator


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


def build_rag_system(vector_store, llm, prompt):
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )


def run_chatbot(vector_store, llm, prompt):
    print("Quran Chatbot is ready. Ask your questions!")
    while True:
        query = input("\nYour Query (type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        try:
            # Retrieve the most relevant context manually
            retriever = vector_store.as_retriever()
            docs = retriever.get_relevant_documents(query)
            if docs:
                # Combine all retrieved document texts as context
                context = "\n".join([doc.page_content for doc in docs])
            else:
                context = "No relevant verse found."

            # Fill the prompt manually with query and context
            filled_prompt = prompt.format(query=query, context=context)
            print("actual prompt: ",filled_prompt)

            # Generate response using LLM
            result = llm(filled_prompt)
            # print(f"Chatbot: {result}")
            response_start = result.find("Response:") + len("Response:")

            # Extract the actual response and strip any leading/trailing whitespace
            response = result[response_start:].strip()

            print(f"Chatbot Response: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

# def remove_repeated_lines(text):
#     lines = text.split("\n")
#     unique_lines = list(dict.fromkeys(lines))  # Removes duplicates while preserving order
#     return "\n".join(unique_lines)

# cleaned_response = remove_repeated_lines(response)
# print(cleaned_response)


# Main Code Execution
if __name__ == "__main__":
    # Load Quran dataset
    quran_df = load_quran_data("main_df.csv")  # Replace with your file path


    vector_store = create_embedding_and_index(quran_df)
    print(vector_store)


    llm = initialize_llm()

    # Create strict prompt
    prompt = create_prompt()

    # Build RAG system
    # rag_system = build_rag_system(vector_store, llm, prompt)

    # Start the chatbot
    run_chatbot(vector_store, llm, prompt)

