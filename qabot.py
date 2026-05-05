# Libraries for Loading Data, Chunking, Embedding, and Vector Databases
# RecursiveCharacterTextSplitter moved to standalone package in LangChain 0.2+
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

#from langchain_community.embeddings import LlamaCppEmbeddings
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama
import gradio as gr
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"

def document_loader(file):
    loader = PyMuPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks
#pip install -U langchain langchain-community langchain-huggingface langchain-text-splitters langsmith
## Vector db
def vector_database(chunks):
    #print(f"==========lengthof the {chunks}")
    #embedding_model = HuggingFaceEmbeddings(model_path='thenlper/gte-large')
    # Using the new package path
    embedding_model = HuggingFaceEmbeddings(
        model_name='thenlper/gte-large',
        # Optional: ensure it runs on your GPU if you have one
        model_kwargs={'device': 'cpu'} 
    )
    print(f"======================================={embedding_model}=================")
    vectordb = Chroma.from_documents(documents=chunks,embedding= embedding_model,persist_directory="./chroma_db")
    return vectordb

model_path = hf_hub_download(
    repo_id=model_name_or_path, 
    filename=model_basename
)

# def watsonx_embedding():
#     mistral_embeddings = LlamaCppEmbeddings(
#         model_path=model_path,
#         n_ctx=2048,  # Context window
#         n_threads=8, # Number of CPU cores to use
#         verbose=False
#     )
#     return mistral_embeddings

## Retriever
def retriever(file):
    print(f"--- Debug: Loading file from {file} ---")
    splits = document_loader(file)
    print(f" ---  Debug: ----- loading splits  ----")
    chunks = text_splitter(splits)
    print(f" ---  Debug: ----- loading chunks  ----")
    vectordb = vector_database(chunks)
    print(f" ---  Debug: ----- loading VectorDB  ----")
    retriever = vectordb.as_retriever()
    return retriever

def get_llm():
    # 3. Initialize LlamaCpp instead of WatsonxLLM
    local_llm = LlamaCpp(
        model_path=model_path,
        temperature=0.5,
        max_tokens=256,
        n_ctx=2048,      # Context window size
        n_threads=8,     # Adjust based on your CPU cores
        n_gpu_layers=0,  # Set to 30+ if you have an NVIDIA GPU/CUDA
        verbose=False    # Set to True to see load logs
    )
    return local_llm

def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=False)
    response = qa.invoke(query)
    return response['result']

rag_application = gr.Interface(
    fn=retriever_qa,
    #allow_flagging="never",
    flagging_mode="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)



rag_application.launch(server_name="0.0.0.0", server_port=7860, share=True)