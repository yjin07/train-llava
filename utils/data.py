from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

llava_model_name_or_path = "my-model/model-01"
llava_processor = AutoProcessor.from_pretrained(llava_model_name_or_path)

@dataclass
class QAIoutput:
    q_input_ids: torch.Tensor
    a_input_ids: torch.Tensor
    pixel_values: torch.Tensor


def build_qai(processor: AutoProcessor, q:str, a:str, image_path:Path) -> dict:
    # ? 1. instruction or input or question
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q}
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # ? 2. Image
    image_file = image_path
    raw_image = Image.open(image_file)

    # ? (1 & 2) together
    inputs = processor(prompt, raw_image, return_tensors="pt")

    # ? 3. Answer
    a_input_ids = processor.tokenizer(
        a, 
        return_tensors="pt",
        padding="longest",
        truncation=True,
    ).input_ids

    # ? 4. Return
    return QAIoutput(
        q_input_ids = inputs["input_ids"],
        a_input_ids = a_input_ids,
        pixel_values = inputs["pixel_values"]
    )



class LlavaDataset(Dataset):
    def __init__(self, data_dir:str):
        super().__init__()

        self.chat_data, self.image_dir = self.build_dataset(data_dir)

    def build_dataset(self, data_dir:str) -> tuple[dict, Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        image_dir = data_dir.joinpath("images")

        chat_data = pd.read_json(chat_file).to_dict(orient='records')

        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    
    def __getitem__(self, idx:int) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[idx]
        human_input = cur_data['conversations'][0]['value']
        gpt_output = cur_data['conversations'][1]['value']

        image_file = self.image_dir.joinpath(cur_data['image'])
        
        return (human_input, gpt_output, image_file)
    


class TrainLlavaCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_IDX: int):
        self.processor = processor
        self.ignore_idx = IGNORE_IDX

    def convert_one_piece(self, 
                        q_input_ids: torch.Tensor,
                        a_input_ids: torch.Tensor) -> dict:
        input_ids= torch.cat([
            q_input_ids, 
            a_input_ids,
            torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1)
            ], dim=1)
        labels = torch.cat([
            torch.full_like(q_input_ids, self.ignore_idx),
            a_input_ids,
            torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1)
            ], dim=1)

        return input_ids, labels


    def __call__(self, features: List) -> dict:
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        max_input_len_list = []

        for feature in features:
            qai_output = build_qai(
                processor=self.processor, 
                q=feature[0], a=feature[1], image_path=feature[2]
            )
            tmp_input_ids, tmp_labels = self.convert_one_piece(
                qai_output.q_input_ids, 
                qai_output.a_input_ids
            )
            max_input_len_list.append(tmp_input_ids.shape[1])
            input_ids_list.append(tmp_input_ids)
            labels_list.append(tmp_labels)
            pixel_values_list.append(qai_output.pixel_values)

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.cat([
            torch.cat([
                torch.full(
                    size=(1, max_input_len - input_ids.shape[1]), 
                    fill_value=self.processor.tokenizer.pad_token_id
                ),
                input_ids
            ], dim=1) for input_ids in input_ids_list
        ], dim=0)

        final_labels = torch.cat([
            torch.cat([
                torch.full(
                    size=(1, max_input_len - labels.shape[1]), 
                    fill_value=self.ignore_idx
                ),
                labels
            ], dim=1) for labels in labels_list
        ], dim=0)

        final_pixel_values = torch.cat(pixel_values_list, dim=0)

        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0 
        

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }


def load_dataset_collator(processor, dataargs: DataArguments):
    llava_dataset = LlavaDataset(
        dataargs.data_path
    )
    data_collator = TrainLlavaCollator(processor, IGNORE_IDX=-100)

    return llava_dataset, data_collator
