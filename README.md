# Quran Chatbot

This project is a Quran-based chatbot that leverages the **Llama-3.2_1B-Instruct** model as a generative model and **all-MiniLM-L6-v2** as an embedding model. The chatbot is built using a **Retrieval-Augmented Generation (RAG)** architecture. It retrieves Quran verses and their explanations from a CSV file provided with the project.

## Features
- **Question-Answering:** Ask any question about Islam or the Quran.
- **Accurate Retrieval:** Provides relevant Quranic verses and their explanations.
- **Streamlit Interface:** A user-friendly web app version.

## Files
- `main_df.csv`: The Quran dataset with verses and explanations.
- `quran_QA.py`: The raw chatbot script.
- `streamlit_version.py`: The Streamlit app version of the chatbot.
- `requirements.txt`: Contains all required Python libraries.

## How to Run

### Step 1: Install Dependencies
Install all the required libraries by running the following command:
```bash
pip install -r requirements.txt
```

### Step 2: Choose Your Version
You have two ways to interact with the chatbot:

#### 1. Raw Chatbot
Run the raw chatbot directly in the terminal:
```bash
python quran_QA.py
```


#### 2. Streamlit Web App
Run the Streamlit version to use the chatbot in a web browser:
```bash
streamlit run streamlit_version.py
```


## Usage
- **Input:** Ask any question related to Islam or the Quran.
- **Output:** The chatbot will retrieve relevant verses and explanations, generating an accurate and concise response.

## Notes
- Ensure `main_df.csv` is in the same directory as the Python scripts.
- For the Streamlit version, open the URL provided after running the command in your web browser.

## License
This project is open-source and free to use. Feel free to modify it as per your needs.

Enjoy exploring knowledge about Islam with this chatbot!
