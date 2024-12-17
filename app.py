import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os

# Page configuration
st.set_page_config(page_title="Restaurant Reservation Bot", page_icon="🍽️")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Setup API keys (replace with st.secrets in production)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

def load_and_process_data():
    """Load and process menu and table data"""
    # Load menu data
    csv1 = CSVLoader("mock_restaurant_menu.csv")
    docs1 = csv1.load()
    
    # Load table data
    csv2 = CSVLoader("table_data (1).csv")
    docs2 = csv2.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs1 = text_splitter.split_documents(docs1)
    split_docs2 = text_splitter.split_documents(docs2)
    
    return split_docs1 + split_docs2

def initialize_llm_chain(docs):
    """Initialize LLM and create retrieval chain"""
    # Initialize LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192"
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a restaurant chatbot. You have information on available tables for reservations and the menu. Answer user questions based on the {context}"),
        ("user", "{input}")
    ])
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY,
        model="models/embedding-001"
    )
    
    # Create document chain
    doc_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create vector store and retriever
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    
    # Create and return retrieval chain
    return create_retrieval_chain(retriever, doc_chain)

def process_audio_input(audio_data):
    """Process audio input and return transcription"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data.read())
            tmp_file.flush()
            
            # Use Gemini for transcription
            audio_file = genai.upload_file(tmp_file.name)
            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content([audio_file, "Transcribe this audio in english exactly word to word"])
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            return result.text
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech and return audio file path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts = gTTS(text=text, lang='en')
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error converting text to speech: {str(e)}")
        return None

# Main application
def main():
    st.title("🎙️ Voice-Enabled Restaurant Reservation System")
    
    # Load data and initialize chain
    docs = load_and_process_data()
    retrieval_chain = initialize_llm_chain(docs)
    
    # Audio input widget
    audio_input = st.audio_input("Speak your request")
    
    if audio_input is not None:
        # Process audio input
        st.info("Processing your request...")
        transcription = process_audio_input(audio_input)
        
        if transcription:
            st.write("You said:", transcription)
            
            # Get bot response
            response = retrieval_chain.invoke({
                "input": transcription,
                "chat_history": st.session_state.chat_history
            })
            
            bot_response = response['answer']
            st.write("Bot response:", bot_response)
            
            # Convert response to speech
            audio_file = text_to_speech(bot_response)
            if audio_file:
                st.audio(audio_file)
                os.unlink(audio_file)  # Clean up audio file
            
            # Update chat history
            st.session_state.chat_history.append({
                "user": transcription,
                "bot": bot_response
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            st.text_area("You:", value=chat["user"], height=50, disabled=True)
            st.text_area("Bot:", value=chat["bot"], height=100, disabled=True)
            st.markdown("---")

if __name__ == "__main__":
    main()
