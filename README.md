# How to fine tune and export to GGML

Grab the data

```
oxen clone ox/Middletownbooks_joke_training
```

Fine tune with LORA

```
time python fine_tune.py ~/Datasets/Middletownbooks_joke_training/data/sft_train_end.parquet results_10-18-2023_end/
```

Run trained model on GPU

```
python run_fine_tuned.py results_10-18-2023_end/final_checkpoint/
```

Merge the LORA weights

```
python merge_lora_model.py results_10-18-2023_end/final_checkpoint/ results_10-18-2023_end/merged_model
```

Convert the merged model from hf to ggml

```
python ~/Code/3rdParty/strutive07/llama.cpp/convert.py results_10-18-2023_end/merged_model/ --outtype f16 --outfile results_10-18-2023_end/merged.bin --vocab-dir meta-llama/Llama-2-7b-hf --vocabtype hf
```

Run the ggml model on CPU

```
python run_on_cpu.py --model results_10-18-2023_end/merged.bin --prompt prompts/joke_prompt.txt
```

Quantize the model to q8_0

```
~/Code/3rdParty/strutive07/llama.cpp/build/bin/quantize results_10-18-2023_end/merged.bin results/merged_ggml_q8_0.bin q8_0
```

Run the quantized model

```
python run_on_cpu.py --model results_10-18-2023_end/merged_ggml_q8_0.bin --prompt prompts/joke_prompt.txt
```

Quantize the model to q4_0

```
~/Code/3rdParty/strutive07/llama.cpp/build/bin/quantize results_10-18-2023_end/merged.bin results/merged_ggml_q4_0.bin q4_0
```

Run the quantized model

```
python run_on_cpu.py --model results_10-18-2023_end/merged_ggml_q8_0.bin --prompt prompts/joke_prompt.txt
```