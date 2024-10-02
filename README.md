# Mutual-Exemplar
This is the official implementation in PyTorch for Mutual Exemplar: Learning Representations by Maximizing Mutual Information Across Views for Medical Image Segmentation

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

## Usage

- **Training:**

1) Use different preprocessing methods for each network.  
2) Update the pseudo-labels after both networks have trained on the entire dataset once.  
3) Slightly modify the loss function in the source code based on the pseudo-labels.
