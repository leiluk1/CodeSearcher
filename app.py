import streamlit as st
from tests.ui_test import get_nearest_code_from_model

languages = ["SQL", "C#", "C", "Python", "Java"]

# Define the available models
models = {
    "seq2seq": {
        "SQL": {
            "prefix": "checkpoints/codet5p-220m-seq2seq/prefix-sql/",
            "ia3": "checkpoints/codet5p-220m-seq2seq/ia3-sql/",
            "lora": "checkpoints/codet5p-220m-seq2seq/lora-sql/",
        },
        "Csharp": {
            "adalora": "checkpoints/codet5p-220m-seq2seq/adalora-csharp/",
            "ia3": "checkpoints/codet5p-220m-seq2seq/ia3-csharp/",
            "lora": "checkpoints/codet5p-220m-seq2seq/lora-csharp/",
            "prefix": "checkpoints/codet5p-220m-seq2seq/prefix-csharp/",
        },
        "C++": {
            "ia3": "checkpoints/codet5p-220m-seq2seq/ia3-cpp/",
            "lora": "checkpoints/codet5p-220m-seq2seq/lora-cpp/",
            "prefix": "checkpoints/codet5p-220m-seq2seq/prefix-cpp/",
        },
        "Python": {
            "lora": "checkpoints/codet5p-220m-seq2seq/lora-python/",
            "prefix": "checkpoints/codet5p-220m-seq2seq/prefix-python/",
        },
    },
    "embeddings": {
           "SQL": {
                "adalora": "checkpoints/codet5p-110m-embedding/adalora-sql",
                "ia3": "checkpoints/codet5p-110m-embedding/ia3-sql",
                "lora": "checkpoints/codet5p-110m-embedding/lora-sql",
                "prompt": "checkpoints/codet5p-110m-embedding/prompt-sql",
            },
            "Csharp": {
                "adalora": "checkpoints/codet5p-110m-embedding/adalora-csharp",
                "ia3": "checkpoints/codet5p-110m-embedding/ia3-csharp",
                "lora": "checkpoints/codet5p-110m-embedding/lora-csharp",
                "prompt": "checkpoints/codet5p-110m-embedding/prompt-csharp",
            },
            "C++": {
                "adalora": "checkpoints/codet5p-110m-embedding/adalora-cpp",
                "ia3": "checkpoints/codet5p-110m-embedding/ia3-cpp",
                "lora": "checkpoints/codet5p-110m-embedding/lora-cpp",
                "prompt": "checkpoints/codet5p-110m-embedding/prompt-cpp",
            },
            "Python": {
                "adalora": "checkpoints/codet5p-110m-embedding/adalora-python",
                "ia3": "checkpoints/codet5p-110m-embedding/ia3-python",
            },
    
    }
}

# Streamlit app
def main():

    # Title and model selection
    st.title("Model Selection")
    selected_language = st.selectbox("Choose a language", languages)
    selected_model_type = st.selectbox("Choose a model type", ["seq2seq", "embeddings"])

    # Get available models based on language and model type
    available_models = models[selected_model_type][selected_language]
    selected_model = st.selectbox("Choose a model", list(available_models.keys()))

    # Input text
    input_text = st.text_area("Enter input text")

    # Button to get decoded output
    if st.button("Get Output"):
        model_path = available_models[selected_model]
        decoded_output = get_nearest_code_from_model(model_path, input_text, selected_language)
        st.write("Output:")
        st.write(decoded_output)

if __name__ == "__main__":
    main()
