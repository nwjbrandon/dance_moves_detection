# Dance Moves Detection

## Setup Dependencies
### Install Python3.7 And Pip
```
sudo apt install python3.7 python3.7-dev python3-pip
```
### Install Virtualenv
```
python3.7 -m pip install virtualenv
python3.7 -m virtualenv venv
source venv/bin/activate
```
### Install Python Dependencies
```
python3.7 -m pip install -r requirements.txt
```

## Notebooks
### HAPT 
Refer to [link](notebooks/hapt/README.md) to understand the model selection and evaluation process on HAPT dataset for detecting dance moves in this project.
### HAPT Quantized
Refer to [link](notebooks/hapt_quantized/README.md) to understand the accuracy and latency of models quantized using different methods.
