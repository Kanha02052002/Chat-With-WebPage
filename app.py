import streamlit as st
from dotenv import load_dotenv
import os
from fpdf import FPDF  # For PDF generation

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OR_TOKEN = os.getenv("OR_TOKEN")

# Page setup
st.set_page_config(page_title="Intelligent Web Content QA", layout="wide")
st.title("🔎 Intelligent Web Content QA System")

# Sidebar - URLs & Actions
st.sidebar.title("📂 Data Source Configuration")
url_input = st.sidebar.text_area("🌐 Enter URLs (comma-separated)", placeholder="https://example1.com, https://example2.com")
load_data = st.sidebar.button("📥 Load & Process Web Data")
clear = st.sidebar.button("🧹 Reset Application")

st.sidebar.write("---")
st.sidebar.write("In order to export, press CTRL+P and save it as PDF.")
# Reset application
if clear:
    st.session_state.clear()
    st.session_state["url_input"] = ""
    st.rerun()  # This will reset the whole app and inputs as well.

# Initialize session state for conversation history and URLs if not already present
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'urls' not in st.session_state:
    st.session_state.urls = []

# Process data
if load_data:
    if not HF_TOKEN or not OR_TOKEN:
        st.error("🚫 API keys not found. Please check your .env file.")
    elif not url_input.strip():
        st.warning("⚠️ Please enter at least one valid URL.")
    else:
        try:
            # Extract and store URLs
            st.session_state.urls = [u.strip() for u in url_input.split(",") if u.strip()]
            with st.spinner("🔄 Fetching and processing website content..."):
                loader = WebBaseLoader(st.session_state.urls)
                content = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
                chunked_docs = text_splitter.split_documents(content)

                embeddings = HuggingFaceInferenceAPIEmbeddings(
                    api_key=HF_TOKEN,
                    model_name="BAAI/bge-base-en-v1.5"
                )

                vectorstore = Chroma.from_documents(chunked_docs, embeddings)
                retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

                st.session_state["retriever"] = retriever
                st.success("✅ Data successfully embedded and ready for querying.")
        except Exception as e:
            st.error(f"❌ Error during processing: {e}")

# Main area – Query input
st.markdown("### 💬 Ask a Question About the Website Content")

query = st.text_input("Type your question below:", placeholder="Ask anything")

# Store the query in session state to maintain across sessions
if query:
    # Append user query to conversation history
    st.session_state.conversation_history.append({"role": "user", "content": query})

    if "retriever" in st.session_state:
        with st.spinner("💬 Generating response..."):
            retriever = st.session_state["retriever"]
            relevant_docs = retriever.get_relevant_documents(query)

            prompt = f"""
            <|system|>>
            You are a helpful and knowledgeable AI assistant.

            Your task is to provide a clear, concise, and accurate response based strictly on the retrieved web content provided by the system. If the answer to the question is not present or cannot be inferred from the content, respond with: "I don't know based on the available data."

            Please structure your response using the following format:
            1. **Answer**: A direct and informative answer to the user's question.
            2. **Supporting Info**: A short explanation or evidence from the content.
            3. **Source Summary**: If applicable, mention where (which section or page) the information appears.

            Only include relevant information from the retrieved content.

            </s>
            <|user|>
            {query}
            </s>
            """

            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OR_TOKEN,
                )
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://yourdomain.com",  # Optional
                        "X-Title": "WebContentQA",  # Optional
                    },
                    model="nvidia/llama-3.1-nemotron-nano-8b-v1:free",
                    messages=[{"role": "user", "content": prompt}]
                )

                response = completion.choices[0].message.content
                st.session_state.conversation_history.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"❌ Error during OpenRouter completion: {e}")

# Display conversation history in reverse order (latest message at the top)
if st.session_state.conversation_history:
    # Display conversation history starting from the second last item
    conversation_pairs = []
    for i in range(0, len(st.session_state.conversation_history), 2):
        # Pair user query and assistant response together
        if i+1 < len(st.session_state.conversation_history):
            user_message = st.session_state.conversation_history[i]
            assistant_message = st.session_state.conversation_history[i+1]
            conversation_pairs.append((user_message, assistant_message))
    
    # Reverse the order so that latest Q&A is displayed at the top
    for user_message, assistant_message in reversed(conversation_pairs):
        st.chat_message("user").markdown(user_message["content"])
        st.chat_message("assistant").markdown(assistant_message["content"])

