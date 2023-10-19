from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
import sys
import torch
from prompt_toolkit import prompt


device_map = {"": 0}
output_dir = sys.argv[1]
base_model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map=device_map, torch_dtype=torch.bfloat16)

context = "You are a humorous chatbot with a charismatically obstreperous yet empathetic and compassionate personality, similar to M*A*S*H's Hawkeye or Patch Adams. Given the joke setup, complete it with a punchline."


while 1:
    question = prompt('> ')



    # text = "### Instruction:\nAnswer the question below\n\n### Input:\n" + question + "\n\n### Response:\n"
    
    text = context + "\n\n" + question + "\n"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(result)
    print("-----")