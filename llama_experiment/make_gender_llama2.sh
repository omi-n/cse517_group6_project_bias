#!/bin/bash

python dump_outputs.py --output_path ./results/gender/female.json --add_sex F --model "meta-llama/Llama-2-70b-chat-hf"

python dump_outputs.py --output_path ./results/gender/male.json --add_sex M --model "meta-llama/Llama-2-70b-chat-hf"