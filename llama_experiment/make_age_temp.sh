#!/bin/bash

# 10,15,18,21,25,30,40,50,60,70
AGES=(10 15 18 21 25 30 40 50 60 70)

for age in "${AGES[@]}"
do
    python dump_outputs.py --output_path ./results/temperature/age/${age}.json --add_age $age --temperature 0.7
done