import logging
import os

import transformers
from transformers import Trainer

from utils.arguments import DataArguments, ModelArguments, TrainingArguments
from utils.data import load_dataset_collator
from utils.model import load_model_processor

logger = logging.getLogger(__name__)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, processor = load_model_processor(model_args)
    train_dataset, data_collator = load_dataset_collator(processor, data_args)

    # 检查 checkpoint_path 是否存在
    checkpoint_path = training_args.resume_from_checkpoint
    if checkpoint_path and not os.path.exists(os.path.join(checkpoint_path, "trainer_state.json")):
        print(f"Checkpoint {checkpoint_path} does not exist or is incomplete. Starting from scratch.")
        checkpoint_path = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()