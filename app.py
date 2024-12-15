import streamlit as st
import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

def translate_role_for_streamlit(role):
    """
    Translate message roles to Streamlit chat roles
    """
    role_mapping = {
        "assistant": "assistant",
        "user": "user",
        "system": "system"
    }
    return role_mapping.get(role, "user")

class SentimentAwareChatbot:
    def __init__(self):
        # Initialize Hugging Face Sentiment Model
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Initialize Groq LLM for response generation
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="mixtral-8x7b-32768"
        )
        
        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Sentiment response templates
        self.sentiment_intros = {
            'Positive': [
                "Great! Here's a helpful response:",
                "Wonderful! Let me help you with that:",
                "Sounds good! Here's what I think:"
            ],
            'Neutral': [
                "I'll help you with that:",
                "Let me provide some information:",
                "Here's a response to your query:"
            ],
            'Negative': [
                "I understand. Let me help you:",
                "I'm here to assist. Here's a response:",
                "I'll do my best to help:"
            ]
        }

    def analyze_sentiment(self, text):
        """
        Analyze sentiment using CardiffNLP RoBERTa model.
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)
        
        # Get predicted label
        predicted_label_idx = probabilities.argmax().item()
        label_names = ['Negative', 'Neutral', 'Positive']
        sentiment = label_names[predicted_label_idx]
        
        # Get confidence score
        confidence = probabilities[0][predicted_label_idx].item()
        
        return sentiment, confidence

    def generate_response(self, user_input, sentiment):
        """
        Generate a response using Groq LLM with sentiment-aware context and chat history.
        """
        try:
            # Get recent chat history (last 5 messages)
            recent_history = st.session_state.chat_history[-5:] if st.session_state.chat_history else []
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            
            # Create a prompt that includes sentiment context and chat history
            full_prompt = f"""
            Chat History:
            {history_str}

            Context: The user's message has a {sentiment} sentiment.
            User's message: {user_input}

            Provide a helpful, comprehensive response while considering the chat history and sentiment.
            """
            
            # Generate response using LLM
            response = self.llm.invoke(full_prompt)
            
            # Update chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Keep only the last 10 messages
            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history = st.session_state.chat_history[-10:]
            
            return response.content
        except Exception as e:
            return f"I'm processing your message about: {user_input}. (Error: {str(e)})"

    def select_sentiment_intro(self, sentiment):
        """
        Select an appropriate sentiment introduction.
        """
        return np.random.choice(self.sentiment_intros[sentiment])

    def display_chat_history(self):
        """
        Display the chat history using Streamlit's chat message containers
        """
        for message in st.session_state.chat_history:
            with st.chat_message(translate_role_for_streamlit(message["role"])):
                st.markdown(message["content"])

    def run_chatbot(self):
        """
        Main chatbot interface and logic with chat history.
        """
        st.title("Advanced Sentiment-Aware Chatbot")

        # Add clear chat history button in sidebar
        st.sidebar.title("Chat Controls")
        if st.sidebar.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

        # Display existing chat history
        self.display_chat_history()

        # Chat Input using chat_input instead of text_input
        user_input = st.chat_input("Type your message here...")

        if user_input:
            # Analyze Sentiment
            sentiment, confidence = self.analyze_sentiment(user_input)
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Show analysis in an expander
            with st.expander("Message Analysis", expanded=False):
                st.write(f"🧠 Sentiment Detected: {sentiment}")
                st.write(f"🎯 Confidence: {confidence:.2%}")
            
            # Generate Response
            try:
                # Select sentiment-based intro
                sentiment_intro = self.select_sentiment_intro(sentiment)
                
                # Generate comprehensive response
                bot_response = self.generate_response(user_input, sentiment)
                
                # Display response with sentiment context
                with st.chat_message("assistant"):
                    st.markdown(f"{sentiment_intro}\n\n{bot_response}")
            
            except Exception as e:
                st.error(f"An error occurred while generating response: {e}")

def main():
    chatbot = SentimentAwareChatbot()
    chatbot.run_chatbot()

if __name__ == "__main__":
    main()