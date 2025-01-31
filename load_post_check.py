from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer():
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification")
    model = AutoModelForSequenceClassification.from_pretrained("badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification")
    
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = load_model_and_tokenizer()
    print("Model and tokenizer loaded successfully.")
