from transformers import AutoModelForCausalLM, AutoTokenizer

def load_language_model(model_name: str = "distilgpt2"):
    """Load GPT-based language model and tokenizer."""
    lm_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return lm_model, tokenizer

lm_model, tokenizer = load_language_model()
