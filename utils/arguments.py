from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="my-model/model-01")
    train_type: Optional[str] = field(
        default="use_lora",
        metadata={
            "helpe": """
            1: use_lora: Use LoRA for training
            2: full: full model training
            3. freeze_vision: freeze vision model
            """
        },
    )



@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the data directory"},
    )