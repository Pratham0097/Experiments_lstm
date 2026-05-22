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

# 2. Prediction Helper Functions
def generate_multiple_words(model, tokenizer, text, max_sequence_len, words_to_predict=3):
    """Generates a sequence of multiple consecutive words."""
    output_text = text
    
    for _ in range(words_to_predict):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len - 1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                next_word = word
                break
        
        if not next_word:
            break
            
        output_text += " " + next_word
        
    return output_text

def predict_top_k_words(model, tokenizer, text, max_sequence_len, k=3):
    """Finds the top K most probable next words with their confidences."""
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    predicted_probs = model.predict(token_list, verbose=0)[0]
    
    # Get the indices of the top K highest probabilities
    top_indices = np.argsort(predicted_probs)[-k:][::-1]
    
    suggestions = []
    for idx in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                suggestions.append((word, predicted_probs[idx]))
                
    return suggestions

# 3. Streamlit UI Layout
st.set_page_config(page_title="Next Word Predictor", page_icon="✍️", layout="centered")

st.title("✍️ Advanced Next Word Predictor")
st.write("An LSTM language model trained on Shakespeare's *Hamlet* to predict and generate text structures.")

st.divider()

# Sidebar Configurations
st.sidebar.header("🔧 Model Settings")
app_mode = st.sidebar.selectbox(
    "Choose Prediction Mode:", 
    ["Multi-Word Generation", "Top 3 Word Suggestions"]
)

MAX_SEQUENCE_LEN = 15  # Matches your model's preprocessing architecture

if app_mode == "Multi-Word Generation":
    num_words = st.sidebar.slider("Words to generate:", min_value=1, max_value=10, value=3)

# Main input window
input_text = st.text_input("Enter your starting phrase:", "To be or not to")

# Execution block
if st.button("Run Prediction", type="primary"):
    if input_text.strip() == "":
        st.warning("Please enter a sequence or phrase first!")
    else:
        if app_mode == "Multi-Word Generation":
            with st.spinner("Generating sequence..."):
                result = generate_multiple_words(model, tokenizer, input_text, MAX_SEQUENCE_LEN, num_words)
            st.success("### Generated Sequence:")
            st.markdown(f"*{result}*")
            
        elif app_mode == "Top 3 Word Suggestions":
            with st.spinner("Analyzing possibilities..."):
                predictions = predict_top_k_words(model, tokenizer, input_text, MAX_SEQUENCE_LEN, k=3)
            
            st.success("### Top Word Options:")
            for i, (word, prob) in enumerate(predictions):
                # Displays the word and a visual progress bar for probability
                st.write(f"**{i+1}. {word}** — *Confidence: {prob*100:.2f}%*")
                st.progress(float(prob))

st.divider()
st.caption("Built with Keras, TensorFlow, and Streamlit.")
