import os
import json
import datetime
import pandas as pd
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# STREAMLIT CONFIGURATION
# -------------------------------
st.set_page_config(page_title="SmartChat NLP", page_icon="💬")

# SSL and NLTK SETUP
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt', quiet=True)

# -------------------------------
# CONSTANTS
# -------------------------------
INTENTS_FILE = "intents.json"
HISTORY_FILE = "chat_history.csv"

# -------------------------------
# INITIALIZE SESSION STATE
# -------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# -------------------------------
# FUNCTION: LOAD INTENTS
# -------------------------------
def load_intents():
    """Load intents from JSON file with error handling."""
    try:
        with open(INTENTS_FILE, "r", encoding="utf-8") as file:
            intents = json.load(file)
            if not intents.get('intents'):
                st.error("The 'intents.json' file is empty or improperly formatted.")
                st.stop()
            return intents
    except FileNotFoundError:
        st.error("The 'intents.json' file is missing. Make sure it exists in the project directory.")
        return {"intents": []}
    except json.JSONDecodeError:
        st.error("Invalid JSON format in 'intents.json'. Please fix the file.")
        return {"intents": []}

intents = load_intents()

# -------------------------------
# FUNCTION: INITIALIZE MODELS
# -------------------------------
@st.cache_resource
def initialize_models(_intents):
    """Initialize and train the TfidfVectorizer and LogisticRegression model."""
    tags = []
    patterns = []
    for intent in _intents['intents']:
        tags.extend([intent['tag']] * len(intent['patterns']))
        patterns.extend([pattern.lower() for pattern in intent['patterns']])

    if not tags or not patterns:
        st.error("No training data found in 'intents.json'.")
        st.stop()

    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    model = LogisticRegression(random_state=0, max_iter=1000)
    X = vectorizer.fit_transform(patterns)
    model.fit(X, tags)
    return vectorizer, model

vectorizer, clf = initialize_models(intents)

# -------------------------------
# FUNCTION: CHATBOT RESPONSE
# -------------------------------
def chatbot_response(user_input):
    """Generate a chatbot response for a given user input."""
    try:
        fallback_responses = [
            "I'm not sure I understand. Could you rephrase?",
            "Sorry, I didn't catch that. Could you explain further?",
            "I'm still learning. Please provide more details."
        ]
        input_vec = vectorizer.transform([user_input.lower()])
        predicted_tag = clf.predict(input_vec)[0]

        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['responses'])

        # Fallback response
        fallback = next((intent for intent in intents['intents'] if intent['tag'] == "fallback"), None)
        return random.choice(fallback['responses']) if fallback else random.choice(fallback_responses)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Oops! Something went wrong."

# -------------------------------
# FUNCTION: SAVE CONVERSATION
# -------------------------------
def save_conversation(user_input, response):
    """Save the conversation to a CSV file."""
    timestamp = datetime.datetime.now().isoformat()
    new_entry = pd.DataFrame([{
        "User": user_input,
        "Response": response,
        "Timestamp": timestamp
    }])
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        updated_df = pd.concat([history_df, new_entry], ignore_index=True)
    else:
        updated_df = new_entry
    updated_df.to_csv(HISTORY_FILE, index=False)

# -------------------------------
# FUNCTION: CLEAR CHAT HISTORY
# -------------------------------
def clear_chat_history():
    """Clear the chat history in session state and delete the CSV file."""
    st.session_state.history = []
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.rerun()  # Add automatic rerun after clearing

# -------------------------------
# FUNCTION: CHAT INTERFACE
# -------------------------------
def chat_interface():
    """Display the chat interface, render chat history, and handle user inputs."""
    st.subheader("💬 Chat with SmartChat!")

    # Display chat history
    with st.container():
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                st.caption(f"({datetime.datetime.fromisoformat(msg['timestamp']).strftime('%Y-%m-%d %H:%M:%S')})")

    # Handle user input
    if user_input := st.chat_input("Type your message here..."):
        response = chatbot_response(user_input)
        timestamp = datetime.datetime.now().isoformat()

        # Update chat history
        st.session_state.history.extend([
            {"role": "user", "content": user_input, "timestamp": timestamp},
            {"role": "assistant", "content": response, "timestamp": timestamp}
        ])

        # Save conversation
        save_conversation(user_input, response)
        st.rerun()  # Add automatic rerun after new message

# -------------------------------
# FUNCTION: DISPLAY CHAT HISTORY
# -------------------------------
def display_chat_history():
    """Display the conversation history from the CSV file."""
    st.subheader("📜 Conversation History")
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        st.dataframe(
            history_df,
            use_container_width=True,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm:ss"),
            },
        )
        st.write(f"Total Conversations: {len(history_df)}")
    else:
        st.info("No conversation history is available.")

# -------------------------------
# FUNCTION: DISPLAY ABOUT SECTION
# -------------------------------
def display_about_section():
    """Show the "About" section with information about the chatbot."""
    st.subheader("ℹ️ About SmartChat")
    st.markdown("""
    ### 🤖 SmartChat: Conversational AI
    SmartChat is designed to provide dynamic and engaging chatbot interactions powered by **Natural Language Processing** and **Machine Learning**.

    **Key Features**:
    - Multi-turn conversations using trained intent classification.
    - Real-time interaction with session-based memory.
    - Easy navigation and history tracking.

    **Built With**:
    - 🐍 Python
    - 📚 NLTK for text tokenization and processing.
    - 📊 Scikit-learn for machine learning.
    - 🖥️ Streamlit for interactive UI.

    Developed with ❤️ by **Animesh Agarkar** | Version **2.1**.
    """)

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    """Main function to drive the chatbot application."""
    st.title("💬 SmartChat: Your Conversational AI Companion")
    st.markdown("Start chatting with an **AI-powered virtual assistant** that's here to make your experience enjoyable!")

    # Sidebar Navigation
    with st.sidebar:
        st.header("Navigation")
        menu = st.radio("Navigate to:", ["💬 Chat", "📜 History", "ℹ️ About"])
        st.button("🧹 Clear History", on_click=clear_chat_history)

    # Render selected menu
    if menu == "💬 Chat":
        chat_interface()
    elif menu == "📜 History":
        display_chat_history()
    elif menu == "ℹ️ About":
        display_about_section()

if __name__ == '__main__':
    main()