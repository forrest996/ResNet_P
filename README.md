# A stage-level network parallelization method based on depth decomposition. [None]

This is the official Pytorch/PytorchLightning implementation of the paper: <br/>
> [**A stage-level network parallelization method based on depth decomposition.**](https:)      
> Zuming Wua,Yunwei Zhanga, 
> *None*
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
git clone https://github.com/forrest996/ResNet_P.git
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Training
Training on CIFER:
```
python DDP_train.py -c './cfg/wideresnet.yaml'
```

### 4. Measure the latency
To measure the latency on CPU/ARM and throughput on GPU, run
```
python ./tools/Fps_Latency.py 
```

## Citation
If you find this repository helpful, please consider citing:
```
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```
