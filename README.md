# Blog Generation Project

This project uses large language models (LLMs) to generate high-quality blog content with a Streamlit web interface. It supports multiple efficient open-source models and provides a simple way to download and use them.
Check out this video to understand how my project works:


## Main Files

### 1. app.py
This is the main Streamlit application. It provides a user interface for generating blogs, selecting the model, customizing the prompt, and downloading the result. It supports multiple models (Llama 2, Mistral, Phi-3, Gemma, TinyLlama) and includes advanced options for SEO, references, and performance.

### 2. download_models.py
A utility script to download alternative LLM models from Hugging Face. Run this script to fetch additional models for use in the app. Models are saved in the `models/` directory and are selectable in the app UI.

### 3. requirements.txt
Lists all required Python packages for the project, including Streamlit, LangChain, CTransformers, Hugging Face Hub, and others. Install dependencies with:

```powershell
pip install -r requirements.txt
```

---

For details on each file, see their respective source code in this repository.
