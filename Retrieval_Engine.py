# retrieval_engine.py

# Import necessary libraries
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Define Default Embedding Function class for retriever
class DefChromaEF(Embeddings):
    def __init__(self, ef):
        self.ef = ef

    def embed_documents(self, texts):
        return self.ef(texts)

    def embed_query(self, query):
        return self.ef([query])[0]

# Setup LLM
def setup_llm(model_path):
    return CTransformers(
        model=model_path,
        model_type="llama",
        lib="avx2"
    )

# Create a prompt template
def create_prompt_template():
    prompt_template_rig = """
    Context:

    {context}

    Task:

    Based on the above context, generate a partial response. If the context doesn’t contain enough information, state 'insufficient data'.
    """
    return PromptTemplate(template=prompt_template_rig, input_variables=['context', 'question'])

# Get a single response using RetrievalQA
def get_response(query, llm, retriever, prompt):
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    return response['result']

# Get interleaved responses (multi-chunk)
def get_interleaved_response(query, retriever, llm, prompt):
    retrieved_docs = retriever.get_relevant_documents(query)
    full_answer = ""

    for idx, doc in enumerate(retrieved_docs):
        chain_type_kwargs = {"prompt": prompt}
        qa_rig = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
        response = qa_rig(query)
        result = response["result"]
        
        print(f"Partial Result from Chunk {idx + 1}:\n", result, "\n")
        full_answer += result + "\n\n"
    
    print('#### The Full Answer is ########### ', full_answer, '################')
    return full_answer








# Main function for retrieval
def main():
    # Initialize Chroma client (replace with actual client instance)
    chroma_client = None  # Define your Chroma client heres
    collection_name = "RIG"
    
    # Setup Langchain Chroma retriever
    langchain_chroma = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=DefChromaEF(embedding_functions.DefaultEmbeddingFunction())
    )
    
    retriever = langchain_chroma.as_retriever(search_kwargs={"k": 2})
    
    # Setup LLM
    local_llm_path = os.getenv('local_llm_path')
    llm = setup_llm(local_llm_path)
    
    # Create prompt template
    prompt = create_prompt_template()
    
    # Example query and response
    query = 'What is the patient taking for pain relief'
    response = get_response(query, llm, retriever, prompt)
    print("Response:", response)
    
    # Get interleaved responses if needed
    full_response = get_interleaved_response(query, retriever, llm, prompt)
    print("Full Interleaved Response:", full_response)

# Run the retrieval engine
if __name__ == "__main__":
    main()
