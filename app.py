import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key from Streamlit secrets or environment
def get_api_key():
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        return st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    except:
        # Fallback to environment variable
        return os.getenv("OPENROUTER_API_KEY")

api_key = get_api_key()

if not api_key:
    st.error("❌ API Key not found! Please set OPENROUTER_API_KEY in Streamlit secrets or .env file")
    st.stop()

# Setup embeddings with OpenRouter
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key
)

# Cache the vector store loading for performance
@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

def main():
    st.title("🔍 RAG Document Query System")
    st.markdown("Ask questions about your company documents (Google, Microsoft, NVIDIA, SpaceX, Tesla)")

    # Sidebar with info
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    This app uses Retrieval-Augmented Generation (RAG) to search through company documents.

    **Documents loaded:**
    - Google.txt
    - Microsoft.txt
    - NVIDIA.txt
    - SpaceX.txt
    - Tesla.txt
    """)

    # Load vector store
    try:
        db = load_vectorstore()
        doc_count = db._collection.count()
        st.sidebar.success(f"✅ Vector store loaded with {doc_count} document chunks")
    except Exception as e:
        st.error(f"❌ Error loading vector store: {e}")
        return

    # Main query interface
    st.header("Ask Your Question")

    # Text input for query
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., How much did Microsoft pay to acquire GitHub?",
        help="Ask questions about the companies in your documents"
    )

    # Submit button
    if st.button("🔎 Search Documents", type="primary"):
        if not query.strip():
            st.warning("⚠️ Please enter a question first")
            return

        with st.spinner("🔄 Searching documents..."):
            try:
                # Create retriever and search
                retriever = db.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(query)

                # Display results
                st.success(f"✅ Found {len(docs)} relevant document chunks")

                if docs:
                    st.subheader("📄 Relevant Documents:")

                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get('source', 'Unknown').replace('docs\\', '').replace('.txt', '')
                        with st.expander(f"📋 Document {i} - {source}", expanded=(i==1)):
                            st.markdown(f"**Source:** {source}")
                            st.markdown(f"**Content length:** {len(doc.page_content)} characters")
                            st.text_area(
                                "Content:",
                                doc.page_content,
                                height=200,
                                key=f"doc_{i}"
                            )
                else:
                    st.info("ℹ️ No relevant documents found for your query. Try rephrasing your question.")

            except Exception as e:
                st.error(f"❌ Error during search: {e}")

    # Footer
    st.markdown("---")
    st.markdown("*Built with LangChain, ChromaDB, OpenRouter, and Streamlit*")

if __name__ == "__main__":
    main()