import streamlit as st
from tests.ui_test import get_decoded_text_from_model

languages = ["C#", "C", "SQL", "Python", "Java"]

# Define the available models
models = {
    "adalora-c#": "checkpoints/codet5p-220m-seq2seq/adalora-csharp/",
    "ia3-c": "checkpoints/codet5p-220m-seq2seq/ia3-c/",
    "ia3-c#": "checkpoints/codet5p-220m-seq2seq/ia3-csharp/",
    "ia3-sql": "checkpoints/codet5p-220m-seq2seq/ia3-sql/",
    "lora-c": "checkpoints/codet5p-220m-seq2seq/lora-c/",
    "lora-c#": "checkpoints/codet5p-220m-seq2seq/lora-csharp/",
    "lora-python": "checkpoints/codet5p-220m-seq2seq/lora-python/",
    "lora-sql": "checkpoints/codet5p-220m-seq2seq/lora-sql/",
    "prefix-c": "checkpoints/codet5p-220m-seq2seq/prefix-c/",
    "prefix-c#": "checkpoints/codet5p-220m-seq2seq/prefix-csharp/",
    "prefix-python": "checkpoints/codet5p-220m-seq2seq/prefix-python/",
    "prefix-sql": "checkpoints/codet5p-220m-seq2seq/prefix-sql/",
}

# Streamlit app
def main():

    # Title and model selection
    st.title("Model Selection")
    selected_model = st.selectbox("Choose a model", list(models.keys()))

    # Language selection
    selected_language = st.selectbox("Choose a language", languages)


    # Input text
    input_text = st.text_area("Enter input text")

    # Button to get decoded output
    if st.button("Get Output"):
        model_path = models[selected_model]
        decoded_output = get_decoded_text_from_model(model_path, input_text, selected_language)
        st.write("Decoded Output:")
        st.write(decoded_output)

if __name__ == "__main__":
    main()

