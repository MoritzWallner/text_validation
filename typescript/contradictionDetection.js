const huggingface = require("@huggingface/inference");

// Hugging Face API details
const API_URL =
  "https://api-inference.huggingface.co/models/facebook/bart-large-mnli";
const API_TOKEN = "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";

// Function to detect contradiction
async function detectContradiction() {
  const inference = new huggingface.HfInference(
    "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"
  );
  const model = "facebook/bart-large-mnli";

  const result = await inference.zeroShotClassification({
    model: model,
    inputs: "I like your outfit!",
    parameters: {
      // candidate_labels: ["contradiction", "entailment", "neutral"],
      candidate_labels: ["insult", "friendly", "neutral"],
    },
  });

  console.log(result);
}

// Example usage
detectContradiction();
