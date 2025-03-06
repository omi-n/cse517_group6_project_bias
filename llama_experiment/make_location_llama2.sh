#!/bin/bash

# HI = "Hawaii"
# ID = "Idaho"
# MA = "Massachusetts"
# SD = "South Dakota"
# VT = "Vermont"
# WY = "Wyoming"
LOCATIONS=(HI ID MA SD VT WY)

for location in "${LOCATIONS[@]}"
do
    python dump_outputs.py --output_path ./results/location_llama2/${location}.json --add_location $location --model "meta-llama/Llama-2-70b-chat-hf"
done