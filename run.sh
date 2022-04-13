#!/bin/bash

if [ ! -d "facereplacement-env" ]
then
    python3 -m venv "facereplacement-env"  # "env" is the name of the environment here.
    source "facereplacement-env/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install opencv-contrib-python
#    python -m pip install opencv-python
#    python -m pip install imutils
    python -m pip install -r requirements.txt
else
    source "facereplacement-env/bin/activate"
fi

#python video_handler.py --src_img imgs/test2.jpg
# python hanlder/video_handler.py --src_img test1.jpg
