# FYP

This project's purpose is to detect breaches of Covid-19 restrictions and send alerts using a Raspberry-Pi.

## Requirements
Python 3.8

## Getting Started

Create a virtual environment and clone this repository.

Install the required dependencies using the command pip3 install -r requirements.txt

Follow the instructions on https://pjreddie.com/darknet/yolo/ to install darknet and Yolov3.

Move the required Yolov3 files specified in social-distance-detector/yolov3 to this directory.

## Models

Pretrained Yolov3 model is used to detect people.

Haar feature-based cascade classifier used to detect faces.
