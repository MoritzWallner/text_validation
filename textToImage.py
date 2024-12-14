import torch
from diffusers import StableDiffusionPipeline

# Use your Hugging Face token
YOUR_HF_TOKEN = "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"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    use_auth_token=YOUR_HF_TOKEN
)

# Use a GPU if available
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate an image from text
prompt = "A beautiful sunset over a mountain range with a clear sky"
prompt = "JUHUU BikeBox GmbH, FN 558602z, located at Marterbauerstraße 4, 3002 Purkersdorf, is responsible for data processing and acts as the data controller under Article 4(7) of the GDPR. They continuously adapt their privacy policy, ensuring that the latest version is always available to users. For any inquiries or to assert data subject rights, individuals can contact JUHUU BikeBox GmbH Customer Service at privacy@juhuu.app."
prompt = "we will also continuously adapt the privacy policy. However, we ensure that the latest version is always available to you. Who is responsible for data processing? JUHUU BikeBox GmbH, FN 558602z, data processing?JUHUU BikeBox GmbH, FN 558602z, Marterbauerstraße 4/, 3002 Purkersdorf, privacy@juhuu.app, is the data controller within the meaning of Article 4(7) GDPR. What do we mean by personal GmbH Customer Service (Subject: Assertion of data subject rights) Marterbauerstraße 4 3002 Purkersdorf E-mail: privacy@juhuu.app Please enclose the following information with your application: o A officer: JUHUU BikeBox GmbH Marterbauerstraße 4 3002 Purkersdorf E-Mail: privacy@juhuu.app In the following cases and for the following purposes, we collect personal data ourselves in accordance with Data Protection gulation (GDPR) In accordance with the provisions of Articles 12ff GDPR, we would like to inform you about the following topics: JUHUU BikeBox GmbH, FN 558602z, Marterbauerstraße 4/,"
image = pipe(prompt).images[0]

# Save or show the image
image.save("generated_image.png")
image.show()
