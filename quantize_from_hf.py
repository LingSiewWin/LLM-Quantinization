from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Mistral-7B-Instruct-v0.3-GPTQ"  # example of GPTQ quantized model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
