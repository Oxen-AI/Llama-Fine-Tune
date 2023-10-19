from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
)
from trl import SFTTrainer
from transformers import LlamaForCausalLM, LlamaTokenizer
import sys

if len(sys.argv) != 3:
    print("Usage: python fine_tune.py <dataset.jsonl> <results_dir>")
    exit()

dataset_file = sys.argv[1]
output_dir = sys.argv[2]

dataset = load_dataset("parquet", data_files={'train': dataset_file})
# dataset = load_dataset("json", data_files={'train': dataset_file})
dataset = dataset['train']
print(dataset)

base_model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True
)
base_model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
base_model.config.pretraining_tp = 1 

# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model, lora_config = create_peft_config(base_model)


training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    # gradient_accumulation_steps=16,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    # learning_rate=1.41e-5,
    learning_rate=2e-4,
    logging_steps=1,
    # max_steps=10000
)

max_seq_length = 1024

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()

import os
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)