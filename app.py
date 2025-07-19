import streamlit as st
import requests

st.set_page_config(page_title="🧠 Chat with HF Models", layout="centered")
st.title("🤖 Hugging Face Chatbot")
st.caption("Select a model and start chatting!")

# --- Model selection ---
models = {
    "GPT2 (small)": "gpt2",
    "DistilGPT2": "distilgpt2",
    "GPT-Neo 1.3B": "EleutherAI/gpt-neo-1.3B",
    "GPT-J 6B": "EleutherAI/gpt-j-6B",
    "Bloom 560M": "bigscience/bloom-560m",
    "Flan-T5 Base (Instruction)": "google/flan-t5-base"
}

model_choice = st.selectbox("Select a Model", list(models.keys()))
model_id = models[model_choice]

# --- Get Hugging Face token from secrets ---
api_token = st.secrets["huggingface_token"]

# --- Set up session state for chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display previous messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Get user input ---
user_input = st.chat_input("Ask me anything...")

# --- Hugging Face Inference API Call ---
def query_huggingface_api(model, prompt):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.7, "max_new_tokens": 200},
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        output = response.json()
        if isinstance(output, list):
            return output[0].get("generated_text", "").replace(prompt, "").strip()
        return output.get("generated_text", "").strip()
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

# --- Handle new input ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_huggingface_api(model_id, user_input)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
