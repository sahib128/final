from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Function to load and return the model based on the model_name
def load_model(model_name: str):
    return Ollama(model=model_name)  # Instantiate the Ollama model with the chosen model_name

# Function to handle the RAG prompt and get a response from the model
def handle_rag_prompt(query_text: str, context_text: str, model):
    # Create prompt for the model
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Get the response from the model
    response_text = model.invoke(prompt)
    
    return response_text

# Function to handle the general prompt and get a response from the model
def handle_general_prompt(query_text: str, model):
    # Directly use the query text as the prompt for general queries
    response_text = model.invoke(query_text)
    
    return response_text

def query_rag(query_text: str, context_text: str, model_name: str):
    # Load the model
    model = load_model(model_name)
    
    # Handle the RAG prompt with the loaded model
    response = handle_rag_prompt(query_text, context_text, model)
    
    return response

def query_general_model(query_text: str, model_name: str):
    # Load the model
    model = load_model(model_name)
    
    # Handle the general prompt with the loaded model
    response = handle_general_prompt(query_text, model)
    
    return response

def main():
    # Define a test query and context
    test_query = "What is the capital of France?"
    test_context = "The capital of France is Paris."
    
    # Define the model name you want to test
    model_name = "llama3.1"  # Replace with the actual model name you are using
    
    try:
        # Query the model with RAG (retrieval-augmented generation) approach
        print("Testing RAG query...")
        response_rag = query_rag(test_query, test_context, model_name)
        print("RAG response:", response_rag)
        
        # Query the model with a general query approach
        print("Testing general query...")
        response_general = query_general_model(test_query, model_name)
        print("General model response:", response_general)
    
    except Exception as e:
        print("An error occurred:", str(e))

# Run the main function
if __name__ == "__main__":
    main()
