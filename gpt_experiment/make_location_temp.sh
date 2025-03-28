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
    python dump_outputs.py --output_path ./results/temperature/location/${location}.json --add_location $location --temperature 0.7
done