# Import necessary libraries
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain.vectorstores import Chroma
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM
from langchain.chains import RetrievalQA
import gradio as gr


class RAG_Pipeline():

    def __init__(self, embedding_model_id, llm_model_id):
        self.loaded_doc = ""
        self.vectordb = None
        self.embedding = self.embedding_model(embedding_model_id)
        self.llm = self.llm_model(llm_model_id)

    # Define embedding model parameters and credentials
    def embedding_model(self, embedding_model_id):

        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }

        watsonx_embedding = WatsonxEmbeddings(
            model_id=embedding_model_id,
            url="https://us-south.ml.cloud.ibm.com",
            project_id=os.getenv("PROJECT_ID"),
            params=embed_params,
            apikey=os.getenv("API_KEY")
        )

        return watsonx_embedding

    # Define LLM parameters and credentials
    def llm_model(self, llm_model_id):
        parameters = {
            GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
            GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
        }
        
        credentials = {
            "url": "https://us-south.ml.cloud.ibm.com",
            "api_key": os.getenv("API_KEY")
        }
        
        project_id = os.getenv("PROJECT_ID")
        
        model = ModelInference(
            model_id=llm_model_id,
            params=parameters,
            credentials=credentials,
            project_id=project_id
        )
        
        llm = WatsonxLLM(watsonx_model = model)
        
        return llm
    
    # Define my RAG pipeline
    def retriever_qa(self, file, query):
        
        ## Check if new pdf file is given as input. Only then create a new vector database
        if file.name != self.loaded_doc:

            self.loaded_doc = file.name
            
            ### Load the PDF Loader
            loader = PyMuPDFLoader(file)
            data = loader.load()

            ### Text Splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50,
                length_function=len
            )

            chunks = text_splitter.split_documents(data)

            ids = [str(i) for i in range(0, len(chunks))]

            ### Create Chroma Vector Database
            new_vectordb = Chroma.from_documents(chunks, self.embedding, ids=ids)

            self.vectordb = new_vectordb

        ## QA Bot
        qa = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff",
            retriever = self.vectordb.as_retriever(),
            return_source_documents = False
        )

        ## Generated output
        response = qa.invoke(query)["result"]

        return response
  
    
if __name__ == "__main__":

    load_dotenv() # Load environment variables

    rag_pipeline = RAG_Pipeline(embedding_model_id="ibm/slate-125m-english-rtrvr", llm_model_id="mistralai/mistral-small-3-1-24b-instruct-2503")

    # Gradio interface
    rag_application = gr.Interface(
        fn=rag_pipeline.retriever_qa,
        allow_flagging="never",
        inputs=[
            gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
            gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
        ],
        outputs=gr.Textbox(label="Output"),
        title="AI RAG Assistant",
        description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
    )

    rag_application.launch(server_name="0.0.0.0", server_port= 7860)