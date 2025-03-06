#!/bin/bash

# 10,15,18,21,25,30,40,50,60,70
AGES=(10 15 18 21 25 30 40 50 60 70)

for age in "${AGES[@]}"
do
    python dump_outputs.py --output_path ./results/age_llama2/${age}.json --add_age $age --model "meta-llama/Llama-2-70b-chat-hf"
done