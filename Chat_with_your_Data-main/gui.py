import streamlit as st
from processingTxt import split_chunks
import os
from chatbot import query_general_model, query_rag

# Define the handle_query function
def handle_query():
    try:
        # Get the selected model or default to "llama3.1"
        model = selected_model if selected_model else "llama3.1"
        
        # Check if there are text chunks available
        if st.session_state.text_chunks:
            # Join text chunks to form the context
            context_text = ' '.join(chunk.page_content for chunk in st.session_state.text_chunks)
            # Use query_rag function for questions with context
            response = query_rag(st.session_state.query_input, context_text, model)
        else:
            # Use query_general_model function for questions without context
            response = query_general_model(st.session_state.query_input, model)
        
        # Update chat history
        st.session_state.messages.append({"role": "user", "content": st.session_state.query_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.query_input = ""  # Clear input field

    except Exception as e:
        st.error(f"Error: {e}")

st.title("PDF Processor")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Apply external CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.subheader("Settings")
model_options = ["llama3.1", "another_model_1", "another_model_2"]
selected_model = st.sidebar.selectbox("Select Model", model_options, index=model_options.index("llama3.1"))
temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
max_length = st.sidebar.slider('Max Length', min_value=32, max_value=128, value=120, step=8)

# Sidebar for file upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("", type="pdf")

if uploaded_file is not None:
    # Save the uploaded PDF file to a temporary file
    temp_pdf_path = "temp_uploaded_pdf.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Process the PDF file
    try:
        st.session_state.text_chunks = split_chunks(temp_pdf_path)
        
        # Add PDF processing info to chat history
        st.session_state.messages.append({"role": "system", "content": "PDF content has been processed and split into chunks."})

    finally:
        # Remove the temporary file
        os.remove(temp_pdf_path)

# React to user input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.query_input = prompt
    handle_query()
