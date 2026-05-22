import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load the pre-trained LSTM model and Tokenizer
@st.cache_resource
def load_resources():
    # Load the Keras model
    model = load_model('next_word_lstm.h5')
    
    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model, tokenizer

try:
    model, tokenizer = load_resources()
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# 2. Helper function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Tokenize the input string
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Keep only the last (max_sequence_len - 1) tokens if input is too long
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
        
    # Pad the sequences
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Predict probabilities for the next token
    predicted_probs = model.predict(token_list, verbose=0)
    
    # Get the index of the highest probability word
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    
    # Map index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

# 3. Streamlit UI Layout
st.set_page_config(page_title="Next Word Predictor", page_icon="✍️", layout="centered")

st.title("✍️ Next Word Prediction with LSTM")
st.write("This app uses a Long Short-Term Memory (LSTM) model trained on Shakespeare's *Hamlet* to predict the next word in a sequence.")

st.divider()

# Input section
input_text = st.text_input("Enter a phrase to start predicting:", "To be or not to")

# Configuration (Adjust this based on your experiments.ipynb max sequence length)
# Commonly, max_sequence_len is set to the length used during training (e.g., 5, 10, or 20)
MAX_SEQUENCE_LEN = 15  

if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing context..."):
            next_word = predict_next_word(model, tokenizer, input_text, MAX_SEQUENCE_LEN)
            
        if next_word:
            st.success(f"**Predicted Next Word:** `{next_word}`")
            # Show completion sequence
            st.markdown(f"**Completed Phrase:** *{input_text.strip()} **{next_word}***")
        else:
            st.error("Could not predict the next word. Try a different phrase.")

st.divider()
st.caption("Built with Keras, TensorFlow, and Streamlit.")
