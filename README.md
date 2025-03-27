# THGD

[TODO] Needs finish

## Environment installation
This code was tested with PyTorch 2.2, cuda 11.8 and torch_geometrics 2.3.1

```bash
conda create -n THGD python=3.10 -y
pip install torch==2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install torch-scatter==2.1.2

pip install -e .
```

## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
    
## Checkpoints

## Generated samples

We provide the generated samples for some of the models. If you have retrained a model from scratch for which the samples are
not available yet, we would be very happy if you could send them to us!


## Troubleshooting 

`PermissionError: [Errno 13] Permission denied: '/home/vignac/MHdiff/src/analysis/orca/orca'`: You probably did not compile orca.