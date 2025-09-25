"""
Model Downloader for Blog Generator App

This script downloads optimized GGUF models for the blog generation application.
Each model is more efficient than the default Llama 2 7B model.
"""

import os
from huggingface_hub import hf_hub_download
import argparse

# Define models with their repository IDs and filenames
MODELS = {
    "mistral": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "Faster and more capable 7B model with excellent instruction following"
    },
    "phi3": {
        "repo_id": "TheBloke/phi-3-mini-4k-instruct-GGUF",
        "filename": "phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "description": "Microsoft's 2.7B model with great reasoning and natural text generation"
    },
    "gemma": {
        "repo_id": "TheBloke/gemma-2b-it-GGUF",
        "filename": "gemma-2b-it.Q4_K_M.gguf", 
        "description": "Google's efficient 2B parameter model for fast inference"
    },
    "tinyllama": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "description": "Ultra-lightweight 1.1B model for extremely fast generation"
    }
}

def download_model(model_name, models_dir="models"):
    """Download a specific model"""
    if model_name not in MODELS:
        print(f"Error: Model '{model_name}' not found. Available models: {', '.join(MODELS.keys())}")
        return False
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    model_info = MODELS[model_name]
    
    print(f"Downloading {model_name} model: {model_info['description']}")
    print(f"From: {model_info['repo_id']}")
    print(f"File: {model_info['filename']}")
    print("This may take several minutes depending on your internet speed...")
    
    try:
        local_path = hf_hub_download(
            repo_id=model_info['repo_id'],
            filename=model_info['filename'],
            local_dir=models_dir
        )
        print(f"✅ Successfully downloaded {model_name} model to: {local_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {model_name} model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download efficient LLM models for blog generation")
    parser.add_argument(
        "--model", 
        choices=["all"] + list(MODELS.keys()),
        default="all",
        help="Specify which model to download (default: all)"
    )
    args = parser.parse_args()
    
    if args.model == "all":
        print("Downloading all models. This will take some time...")
        for model_name in MODELS.keys():
            success = download_model(model_name)
            print("-" * 50)
        print("Download process complete. Check above for any errors.")
    else:
        success = download_model(args.model)
        if success:
            print("Download process complete.")
        else:
            print("Download process failed.")

if __name__ == "__main__":
    main()