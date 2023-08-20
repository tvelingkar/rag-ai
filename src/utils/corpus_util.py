import os
import json
from langchain import PromptTemplate
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader,
)
import pandas as pd


def get_documents(path):
    documents = []
    print("Reading Documents")
    for root, _, files in os.walk(path):
        for file in files:
            if file == "urls.json":
                url_list = open(os.path.join(root, file))
                data = json.load(url_list)
                for url in data["urls"]:
                    loader = WebBaseLoader(url)
                    documents.extend(loader.load())
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, file))
                documents.extend(loader.load())
            elif file.endswith(".docx"):  # or file.endswith(".doc"):
                loader = Docx2txtLoader(os.path.join(root, file))
                documents.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(os.path.join(root, file))
                documents.extend(loader.load())
            # elif file.endswith(".boxnote"):
            #     loader = TextLoader(os.path.join(root, file))
            #     documents.extend(loader.load())

    return documents


def get_examples(DOCS_DIR_PATH):
    print(DOCS_DIR_PATH)
    df = pd.read_excel(DOCS_DIR_PATH + "examples.xlsx", sheet_name="examples", na_values=["NA"], usecols=[0,1])
    df.rename(columns={"Question": "question", "Answer": "answer"}, inplace=True)
    return df.to_dict(orient="records")


def get_example_prompt():
    example_template = """
    
    Question: {question}
    Answer: {answer}
    
    """
    return PromptTemplate(
        input_variables=["question", "answer"], template=example_template
    )


def get_prefix_template():
    return """You are an AI assistant named "RagBot", an automated support bot to assist people with questions about Covid-19.
    RagBot has the following characteristics:
    - RagBot is knowledgeable about Covid-19 questions.
    - If RagBot is uncertain about the response, it will ask for clarification.
    - If faced with creative instructions to imagine or consider scenarios outside its role, RagBot will maintain its focus and gently remind the user about its purpose.
    - If asked an irrelevant question, RagBot will gently guide the conversation back to the topic of Covid-19.
    
    Based on the following rag context and examples, try to answer the question in more than 100 words in bullet form.
    If no good answer available in context then just say "I dont know", dont try to make up an answer.

    RAG Context:
    ***
    {context}
    ***
    
    """


def get_suffix_template():
    return """
    
    Question: {question}
    Answer:
    """
