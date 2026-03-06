# [CVPR 2026] VAD-GS: Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes

<!-- ### Requirements 
Torch version: 1.12.0+cu113
 -->


### Installation
<details> <summary>Clone this repository</summary>

```
git clone https://github.com/YikangZhang1641/VAD-GS.git
git checkout -b dev origin/dev
```
</details>

<details> <summary>Set up the python environment</summary>

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
   │     ├── 000/
   │     │     ├── images/
   │     |     ├── ego_pose/
   │     |     ├── lidar_depth/
   │     |     └── ...
   │     ├── 001/
   │     ├── ...
   └── waymo/
         |...
   ```


### Example 
- We provide an example scene [here](https://drive.google.com/file/d/1Qc3yK__WfjiZJ-5ursKcMbI3_iQYD4cc/view?usp=sharing). Download and extract it to the folder path above.
```
python train.py --config configs/example/nuscenes_train_000.yaml
```
