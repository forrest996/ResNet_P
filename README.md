# A stage-level network parallelization method based on depth decomposition.

This is the official Pytorch/PytorchLightning implementation of the paper: <br/>
> [**A stage-level network parallelization method based on depth decomposition.**](https:)      
> Zuming Wua,Yunwei Zhanga, 
> *IEEE Access*
> 

---
### 1. Dependency Setup
Create an new conda virtual environment
```
conda create -n ResNet_P python=3.8.10 -y
conda activate ResNet_P
```
Clone this repo and install required packages:
```
git clone https://github.c
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### 2. Dataset prepare
CIFAR:
```
https://www.cs.toronto.edu/~kriz/cifar.html

Download the file and extract it to:
--data
    ---CIFAR-10
        ----data_batch_1
        ----data_batch_2
        ----data_batch_3
        ----data_batch_4
        ----data_batch_5
        ----test_batch
    ---CIFAR-100
        ----meta
        ----test
        ----train
```


### 3. Training
Training on CIFAR:
```
python DDP_train.py -c './cfg/wideresnet.yaml'
```

### 4. Measure the latency
To measure the latency on CPU/ARM and throughput on GPU, run
```
python ./Fps_Latency.py 
```
