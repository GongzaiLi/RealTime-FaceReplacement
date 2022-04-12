#!/bin/bash

if [ ! -d "facereplacement-env" ]
then
    python3 -m venv "facereplacement-env"  # "env" is the name of the environment here.
    source "facereplacement-env/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install opencv-contrib-python==4.5.5.62
    python -m pip install opencv-python
    python -m pip install -r requirements.txt
else
    source "facereplacement-env/bin/activate"
fi

#python video_handler.py --src_img imgs/test7.jpg