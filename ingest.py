import os
import pathlib
from dotenv import load_dotenv


from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

from src.utils.corpus_util import (
    get_documents,
    get_example_prompt,
    get_examples,
    get_prefix_template,
    get_suffix_template,
)

load_dotenv()

APP_NAME = "RAG_AI"
model = "google/flan-t5-base"
condense_model = "google/flan-t5-xl"

PATH = pathlib.Path(__file__).parent.resolve()
DATA_DIR = str(PATH) + os.sep + "data"
DOCS_DIR_PATH = str(PATH) + os.sep + "docs" + os.sep

embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large", model_kwargs={"device": "cpu"}
)

def load_db():
    resync_model = os.getenv("RESYNC_MODEL", "False").lower()
    if resync_model == "true":
        documents = get_documents(DOCS_DIR_PATH)
        text_splitter = RecursiveCharacterTextSplitter(
            # Set custom chunk size
            chunk_size=512,
            chunk_overlap=20,
            # Use length of the text as the size measure
            length_function=len,
        )
        documents = text_splitter.split_documents(documents)
        print("Vectorizing and Persisting Documents")
        vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
        vectordb.save_local(DATA_DIR, APP_NAME)
        vectordb = None
        
    db = FAISS.load_local(DATA_DIR, embeddings, APP_NAME)
        
    return db

def set_prompt():
    examples = get_examples(DOCS_DIR_PATH)
    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
        # This is the list of examples available to select from.
        examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        embeddings,
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        FAISS,
        # This is the number of examples to produce.
        k=2,
        input_keys=["question"],
    )
    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=get_example_prompt(),
        prefix=get_prefix_template(),
        suffix=get_suffix_template(),
        input_variables=["context", "question"],
    )
    
    return prompt

def load_llm():
    llm = HuggingFaceHub(
        repo_id=model, model_kwargs={
            "max_new_tokens": 250,
            "min_new_tokens": 10,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 1,
        }
    )
    return llm

def retrieve_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def generate_answer(question):
    db = load_db()
    llm = load_llm()
    qa_prompt = set_prompt()
    qa_chain = retrieve_qa_chain(llm, qa_prompt, db)
    result = qa_chain({"query": question})
    return {
        "answer": result["result"]
    }
