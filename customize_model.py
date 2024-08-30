from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import LlavaForConditionalGeneration, LlavaConfig


clip_model_name_or_path = "clip-vit-large-patch14-336"
qwen_model_name_or_path = "Qwen1.5-4B-Chat"


clip_model = AutoModel.from_pretrained(clip_model_name_or_path, device_map="cuda:0")
llm_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_name_or_path, device_map="cuda:0"
)


llm_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path)

# * Add and test the image token to the tokenizer
print(llm_tokenizer.encode("<image>"))

# * Get the configuration of the models
vision_config = clip_model.vision_model.config
text_config = llm_model.config

# * Initialize the llava model
configuration = LlavaConfig(vision_config, text_config)
model = LlavaForConditionalGeneration(configuration)

# * Load the weights of the models
model.vision_tower.vision_model = clip_model.vision_model
model.language_model = llm_model

# * Set the pad token id and image token index
model.config.pad_token_id = llm_tokenizer.pad_token_id
model.config.image_token_index = llm_tokenizer.encode("<image>")[0]

# * Save the model
model.save_pretrained('my-model/model-01')
llm_tokenizer.save_pretrained('my-model/model-01')

# * Save the processor
autoprocessor = AutoProcessor.from_pretrained(clip_model_name_or_path)
autoprocessor.save_pretrained('my-model/model-02')

# Note:
# The `preprocessor_config.json` file located in `my-model/model-02` needs to be manually 
# copied and placed into `my-model/model-01` to ensure compatibility between the processor 
# and the customized LLava model.