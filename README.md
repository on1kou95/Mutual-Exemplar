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

- Download kvasir-instrument.zip.
- Move it to the Mutual_Exemplar folder.
- Run python split_dataset.py in that folder.

## Usage

- **Training:**
You can directly run the codes ```Mutual_Exemplar_train.py``` and ```Min_Max_Similarity_train.py``` for the proposed method and the best competing method, respectively.
