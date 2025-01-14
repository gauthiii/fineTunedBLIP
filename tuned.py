import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the fine-tuned processor and model
processor = BlipProcessor.from_pretrained("fine_tuned_blip")
model = BlipForConditionalGeneration.from_pretrained("fine_tuned_blip")

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Path to the new image you want to caption
img_path = "tnj/30.jpg"  # Replace with your image path
image = Image.open(img_path).convert('RGB')

# Preprocess the image
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt", truncation=True).to(device)

# Generate the caption
outputs = model.generate(**inputs, max_length=50)
caption = processor.decode(outputs[0], skip_special_tokens=True)

# Print the generated caption
print("Generated Caption:", caption)
