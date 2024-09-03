from dataclasses import dataclass, field
from typing import Optional

import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="my-model/model-01")
    train_type: Optional[str] = field(
        default="use_lora",
        metadata={
            "help": """
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

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    save_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Number of update steps between two checkpoint saves."},
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "The maximum number of checkpoints to keep. Older checkpoints will be deleted."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint to resume training from."},
    )
    save_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use: 'steps' or 'epoch'."},
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to load the best model found during training at the end of training."},
    )
    evaluation_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use: 'no', 'steps', or 'epoch'."},
    )
    eval_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of update steps between two evaluations."},
    )