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

st.markdown("""
<style>
    /* Main app background - matching Amazon gradient */
    .stApp {
        background: linear-gradient(135deg, 
            #000000 0%, 
            #004D2C 100%);
    }
    
    /* Headers - bolder white */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: 0px;
    }
    
    /* Subheader */
    .subheader {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        opacity: 0.9;
    }
    
    /* Text areas */
    .stTextArea {
        background: linear-gradient(145deg, 
            rgba(0, 51, 30, 0.95) 0%, 
            rgba(0, 77, 44, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #FFFFFF !important;
        font-weight: 500;
    }
    
    /* Success messages */
    .st-emotion-cache-16idsys {
        background: rgba(0, 77, 44, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
        font-weight: 500;
    }
    
    /* Info messages */
    .st-emotion-cache-16r3i8g {
        background: rgba(0, 51, 30, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton button {
        background: #FFFFFF;
        color: #004D2C;
        border: none;
        padding: 12px 24px;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: #F0F0F0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Audio player */
    .stAudio {
        background: rgba(0, 51, 30, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Chat container */
    .chat-container {
        background: linear-gradient(145deg,
            rgba(0, 51, 30, 0.95) 0%,
            rgba(0, 77, 44, 0.95) 100%);
        border-radius: 8px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 15px 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        background: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 3px;
    }
    
    /* Company branding */
    .company-brand {
        text-align: center;
        padding: 15px;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        font-size: 0.9em;
        font-weight: 600;
        letter-spacing: 0.5px;
        position: fixed;
        bottom: 0;
        width: 100%;
        background: rgba(0, 51, 30, 0.95);
    }

    /* Regular text */
    .stMarkdown {
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

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
    ("system", "You are a restaurant chatbot with detailed information about available tables, reservations, and the menu. Respond to user inquiries based on the {context}. When the user greets you, provide a warm and friendly greeting. When the user specifies the number of guests and preferred time, elegantly present all available tables, highlighting their locations with refined and attractive language. Once the user selects a table, confirm the reservation and finalize it—do not ask further questions after that. Additionally, if the user places a food order, guide them through the menu and handle their order efficiently, confirming their selection before concluding the conversation."),
    ("user", "{input}")
])

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
            st.text_area("You:", value=chat["user"], height=70, disabled=True)
            st.text_area("Bot:", value=chat["bot"], height=100, disabled=True)
            st.markdown("---")

if __name__ == "__main__":
    main()
