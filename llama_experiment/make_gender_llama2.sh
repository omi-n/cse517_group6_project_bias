#!/bin/bash

python dump_outputs.py --output_path ./results/gender_llama2/female.json --add_sex F --model "meta-llama/Llama-2-70b-chat-hf"

python dump_outputs.py --output_path ./results/gender_llama2/male.json --add_sex M --model "meta-llama/Llama-2-70b-chat-hf"