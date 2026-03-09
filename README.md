# [CVPR 2026] VAD-GS: Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes
[Project page](https://mias.group/VAD-GS/) |  [paper](https://arxiv.org/abs/2510.09364)



### Installation
<details> <summary>Clone this repository and checkout dev branch</summary>

```
git clone https://github.com/YikangZhang1641/VAD-GS.git
git checkout -b dev origin/dev
```
</details>


<details>
<summary>Build tools</summary>

1. Install [COLMAP](https://colmap.github.io/install.html) (tested version 3.10-dev)
2. Build SIBR_viewer following the tutorial of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)

</details>




<details> <summary>Set up the environment</summary>

```
# Set conda environment
conda create -n vadgs python=3.8
conda activate vadgs

# Install torch (corresponding to your CUDA version)
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install requirements
pip install -r requirements.txt

# Install submodules
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/simple-waymo-open-dataset-reader
pip install ./submodules/MyPropagation
```
</details>





### Datasets 

```
data/
   ├── nuscenes/
   │     ├── raw/
   │     ├── processed_10Hz/
   │     │     ├── mini/
   │     │     │     ├── 000/
   │     │     │     │     ├── images/
   │     |     │     │     ├── ego_pose/
   │     |     │     │     ├── lidar_depth/
   │     |     │     │     └── ...
   │     │     │     ├── 001/
   │     │     │     ├── ...
   └── waymo/
         |...
   ```


### Example 
- We provide a nuScenes example [here](https://drive.google.com/file/d/1suOq_jm3hJt-HyWGTRyY8s58sbM2Z1uz/view?usp=sharing). Download and extract it to the folder path above.

For training:
```
python train.py --config configs/example/nuscenes_train_000.yaml
```

To generate visual outputs:
```
python render.py --config configs/example/nuscenes_train_000.yaml mode evaluate
```

For evaluation:
```
python metrics.py --config configs/example/nuscenes_train_000.yaml
```
