from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
import sys
import torch
from prompt_toolkit import prompt

if len(sys.argv) != 3:
    print("Usage: python merge_lora_model.py <lora_dir> <output_dir>")
    exit()

device_map = {"": 0}
lora_dir = sys.argv[1]
base_model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(lora_dir, device_map=device_map, torch_dtype=torch.bfloat16)


model = model.merge_and_unload()

output_dir = sys.argv[2]
model.save_pretrained(output_dir)