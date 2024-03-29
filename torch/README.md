# CIRCLE

## Dependency 
- CUDA == 10.2 or 11.1 (we only run it on CUDA 10.2 and CUDA 11.1 but it should works for any version of CUDA 10 or CUDA 11)
- torch >= 1.7.1 
- [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
- open3d == 0.14.1
- python >= 3.7
## Training
The network is trained on [MatterPort3d](https://niessner.github.io/Matterport/)
### Data
We provide tools to generate incomplete scenes from MatterPort3d
```
python sample_matterport.py <train/val/test> <sample_rate>
python add_noise.py <train/val/test> <sample_rate> <noise_sigma>
python crop_small.py <train/val/test> <sample_rate> <noise_sigma> 
```
For training data all three step is needed, for test and validation, we just run first two step.
In paper, the sample_rate and the noise_sigma is 0.5 and 0.01, and the provided pretrained model is trained with same setting.
### Train the network
Once the dataset is generated, run the following script to start training. 
```
python train.py <EXPERIMENT_NAME>
```
Path to dataset and config file could be modified in train.py manualy.

## Test
To test the model, we provide code and [weight](https://drive.google.com/file/d/12N0hlYbJFF4wiJGqVeSRxVaOWimr1fp9/view?usp=sharing) for MatterPort3D dataset.
```
python run_matterport.py <EXPERIMENT_NAME> <CHECKPOINT_ITER> <MATTERPORT_SCENE> <MATTERPORT_REGION>
``` 

## Inference-time Optimization
To use differential renderer for inference-time optimization, one can use the following script
```
python dr_matterport.py <MATTERPORT_SCENE> <MATTERPORT_REGION>
```

## Customize Your Own Dataset
To train on your own dataset, you can write your own dataloader providing same output of NoisyOtherMatterPortDataset.

Test on your own dataset without differentiable renderer is easy, change the input in `run_matterport.py` is enough. 

To enable differentiable renderer, you should write your own `get_inputs()` function in `dr_matterport.py`.



## Notes

- The pytorch extensions in `ext/` will be automatically compiled on first run, so expect a delay before training start.
