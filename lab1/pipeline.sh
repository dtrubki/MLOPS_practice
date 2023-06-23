#!/bin/bash
echo start pipeline
pip3 install -r requirements.txt
echo requirements installed

python3 data_creation.py
echo data downloaded

python3 model_preprocessing.py
echo preprocessing end

python3 model_preparation.py
echo model created

python3 model_testing.py
echo pipeline end
