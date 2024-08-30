# Llava Model Training: Crafting a Multimodal AI with Vision and Language Fusion

## Overview

Llava, short for [Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA), is a multimodal model that integrates vision and language understanding, enabling it to process and generate information from both image and text inputs. In this project, we use OpenAI’s `CLIP-ViT-Large-Patch14-336` as the visual component of LLava and `Qwen1.5-4B-Chat` as the language component. This setup allows LLava to effectively handle tasks that require a fusion of visual and language processing. The project focuses on building and training LLava using advanced techniques and custom datasets to optimize its performance.

## Model Building
The LLava model is constructed by integrating two pre-trained models: `CLIP-ViT-Large-Patch14-336` for image processing and `Qwen1.5-4B-Chat` for language processing. Both models can be downloaded from the Hugging Face model hub. Below are the steps to customize and build the LLava model:

1. Modify the `Qwen` tokenizer

    To ensure that the tokenizer recognizes image tokens, modifications need to be made to the `tokenizer_config.json` file of Qwen.
    + Set `<image>` Token ID:

        In the `tokenizer_config.json` file, add the `<image>` token to the `added_tokens_decoder` section with the following configuration:

        ```json
        "151646": {
            "content": "<image>",
            "lstrip": false,
            "normalized": false,
            "rstrip": false,
            "single_word": false,
            "special": true
        }
        ```
    + Add `<image>` to Special Tokens:

        Also, ensure that `<image>` is included in the `additional_special_tokens` list within the `tokenizer_config.json` file like:
        ```json
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<image>"]
        ```

2. Customize the Llava Model:

    Use the `customize_model.py` to customize your LLava model. This script allows you to set up and fine-tune the integration of the visual and language components. Save the customized model in the directory `my-model/model-01`.


## Training Data
The `LLaVA-CC3M-Pretrain-595K` dataset is used for training the LLava model. This dataset provides a comprehensive collection of image-text pairs, essential for multimodal training. The process of constructing the training dataset and preparing it for model training is handled through the utils/data.py file.

The `utils/data.py` file contains all the necessary code for creating the training dataset. It covers key steps, including data loading, preprocessing, and formatting the data into a structure suitable for training, ensuring consistency and efficiency in data preparation.

The `data.py` file also includes the implementation of a custom data collator, which is crucial for batching and processing data correctly during training. The collator handles image and text inputs, ensuring they are properly aligned and fed into the model during each training step.

By encapsulating all dataset creation and collator logic within `utils/data.py`, the training process is streamlined. This setup allows the training pipeline to directly access and utilize the prepared data without additional overhead, making the overall process more efficient and less prone to errors.

This structured approach ensures that the LLava model is trained with well-prepared data, enhancing its ability to learn from complex multimodal inputs and improving overall performance.


## Training strategies

We consider three different strategies to train the Llava model, each tailored to balance performance and computational efficiency based on specific needs:

+ Full Parameter Training (`--train_type full`):
In this approach, all model parameters, including both the vision and language components, are fully trainable. This method allows the model to learn new representations but requires significant computational resources and training time. It is typically used when maximum flexibility in model learning is desired, although it may not always yield the best performance due to the high complexity and risk of overfitting.

+ LoRA (Low-Rank Adaptation) (`--train_type use_lora`):
LoRA is an efficient training strategy that focuses on fine-tuning specific layers of the model by injecting low-rank updates into the original parameter space. This method significantly reduces the number of trainable parameters, making training faster and more resource-efficient. In the context of LLava, LoRA is particularly effective as it selectively updates the language layers while freezing the vision layers, preserving the model's ability to understand visual inputs while optimizing language processing.

+ Freezing the Vision Layer, Full Parameter Training of the Language Layer (`--train_type freeze_vision`):
In this strategy, the visual component is frozen, meaning its weights are not updated during training. Only the language component undergoes full parameter training. This approach balances the computational load by keeping the vision features static while allowing the language layer to be fully adaptable. It is often used when the visual model's pre-trained features are already well-suited to the task, and the focus is on improving the language model's performance.

Each of these strategies provides distinct advantages, depending on the computational constraints and the specific goals of the training. The choice of strategy should be guided by the task requirements, available resources, and desired balance between training efficiency and model flexibility. 

From experience, LoRA tends to yield the best performance across various tasks due to its efficient adaptation of key model components. However, the actual performance may vary depending on the specific task, and it's advisable to experiment with different strategies to find the optimal approach for your needs.

## Training Process

The training process for the LLava model is conducted using a SLURM script, `run.sh`, which manages the distributed training across multiple GPUs. The training utilizes DeepSpeed with a Zero-2 optimization strategy and runs on four A100 GPUs to efficiently handle large-scale model training.