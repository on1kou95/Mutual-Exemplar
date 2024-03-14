# Mutual-Exemplar
This is the official implementation in PyTorch for Mutual Exemplar: Contrastive Co-training for Surgical Tools Segmentation

## Environment

- python==3.6
- packages:
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
```
conda install opencv-python pillow numpy matplotlib
```
- Clone this repository
```
git clone https://github.com/AngeLouCN/Min_Max_Similarity
```
## Data Preparation

- [Kvasir-instrument](https://datasets.simula.no/kvasir-instrument/)

**File structure**
```
|-- data
|   |-- kvasir
|   |   |-- train
|   |   |   |--image
|   |   |   |--mask
|   |   |-- test
|   |   |   |--image
|   |   |   |--mask
|   |-- EndoVis17
|   |   |-- train
|   |   |   |--image
|   |   |   |--mask
|   |   |-- test
|   |   |   |--image
|   |   |   |--mask
......
```

**You can also test on some other public medical image segmentation dataset with above file architecture**

## Usage

- **Training:**
You can change the hyper-parameters like labeled ratio, leanring rate, and e.g. in ```train_mms.py```, and directly run the code.
