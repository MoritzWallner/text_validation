from transformers import pipeline

# Load a pre-trained NLI model from Hugging Face
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

def check_contradiction(premise, hypothesis):
    # Format the input correctly for the NLI model
    inputs = f"{premise} [SEP] {hypothesis}"
    
    # Use the NLI model to classify the relationship between premise and hypothesis
    result = nli_model(inputs)
    
    # Extract the label (entailment, contradiction, or neutral)
    label = result[0]['label']
    score = result[0]['score']
    
    return label, score

# Example usage
# generated_text = "The Eiffel Tower is located in New York."
generated_text = "The Eiffel Tower is in Tianducheng, China."
ground_truth = "The Eiffel Tower is located in Paris."

# Check for contradiction
label, confidence = check_contradiction(generated_text, ground_truth)

print(f"Label: {label}, Confidence: {confidence}")
