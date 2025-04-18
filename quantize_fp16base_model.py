from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
save_dir = "./mistral-7B-quant"

# Load the model in fp16
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype="auto", 
    device_map="auto"
)

# Define quantization config
quant_config = BaseQuantizeConfig(
    bits=4,  # INT4
    group_size=128,
    desc_act=False,
)

# Run quantization
quant_model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=quant_config,
    use_triton=False
)

# Save quantized model
quant_model.save_quantized(save_dir)
tokenizer.save_pretrained(save_dir)
