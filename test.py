from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch


model_name_or_path = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path, device_map=device, torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
url = "data/australia.jpg"
image = Image.open(url)

inputs = processor(text=prompt, images=image, return_tensors="pt")

for temp_key in inputs.keys():
    inputs[temp_key] = inputs[temp_key].to(device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=15)

response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)
