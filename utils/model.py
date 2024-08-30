from .arguments import ModelArguments
import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
import logging

def load_model_processor(modelargs: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path)

    if modelargs.train_type == "use_lora":
        logging.warning("Loading model to LoRA")

        from peft import LoraConfig, get_peft_model

        LORA_R = 32
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=LORA_R,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            module_to_save=["multi_modal_projector"]
        )

        model = get_peft_model(model, config)

    elif modelargs.train_type == "none":
        logging.warning("Train full model")

        pass

    elif modelargs.train_type == "freeze_vision":
        logging.warning("Freeze vision model")

        for param in model.vision_tower.parameters():
            param.requires_grad = False

    # print the number of trainable parameters
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_parameters}")

    return model, processor



