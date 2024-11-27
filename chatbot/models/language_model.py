from transformers import AutoModelForCausalLM, AutoTokenizer

def cache_language_models(model_names):
    for model_name in model_names:
        print(f"Downloading and caching model: {model_name}")
        AutoModelForCausalLM.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        print(f"Cached: {model_name}")

# Example usage
model_list = [
    "distilgpt2",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B"
]


cache_language_models(model_list)

def load_language_model(pretrained_model: str = "distilgpt2"):
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    return model, tokenizer