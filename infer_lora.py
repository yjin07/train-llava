from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from peft import peft_model, PeftModel
from utils.data import LlavaDataset
from PIL import Image

raw_model_name_or_path = "/blue/amolstad/y.jin/train-llava/my-model/model-01"
peft_model_name_or_path = "/blue/amolstad/y.jin/train-llava/Results"

model = LlavaForConditionalGeneration.from_pretrained(raw_model_name_or_path, 
                                                    device_map="cuda:0", 
                                                    torch_dtype=torch.float16)

model = PeftModel.from_pretrained(model, peft_model_name_or_path, adapter_name="peft_v1")
processor = AutoProcessor.from_pretrained(raw_model_name_or_path)
model.eval()
print('ok')


llavadataset = LlavaDataset("/blue/amolstad/y.jin/train-llava/data")
len(llavadataset), llavadataset[10]

testdata = llavadataset[102]
print(len(testdata))
print(testdata[0])
print(testdata[1])
print(testdata[2])

Image.open(testdata[2])


def build_model_input(model, processor, testdata):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": testdata[0]},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(prompt)
    # print("*" * 20)

    image = Image.open(testdata[2])
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    for tk in inputs.keys():
        inputs[tk] = inputs[tk].to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=20)

    generated_ids = [
        oid[len(iids):] for oid, iids in zip(generated_ids, inputs.input_ids)
    ]

    # gen_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    gen_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    return gen_text


build_model_input(model, processor, testdata)


model = model.merge_and_unload()
model.save_pretrained("output_model_lora_merge_001")
processor.save_pretrained("output_model_lora_merge_001")