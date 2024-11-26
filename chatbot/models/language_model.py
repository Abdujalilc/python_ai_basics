from transformers import AutoModelForCausalLM, AutoTokenizer

def load_language_model(pretrained_model="distilgpt2"):
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    return model, tokenizer

lm_model, tokenizer = load_language_model()
