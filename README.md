# SimCLR
This code reproduction of SimCLR framework in PyTorch implementation

Original [paper](https://arxiv.org/pdf/2002.05709.pdf) and [code](https://github.com/google-research/simclr) were provided

This is my review of [paper](https://breezy-perfume-dec.notion.site/A-Simple-Framework-for-Contrastive-Learning-of-Visual-Representations-5d223a6d07304b1ca6c708f8d940e9cd) and [code](https://breezy-perfume-dec.notion.site/SimCLR-Code-Review-e086668735814bab977c3c4cc8f7e661)

<p align=center><img src = https://user-images.githubusercontent.com/60006301/197710408-b85b2551-e8a5-4e21-a413-6ce22ac8d976.png width=500></p>

## Installation
```
$ conda env create --name simclr --file env.yaml
$ conda activate simclr
$ python3 run.py
```
download path file [here](https://drive.google.com/file/d/16lxETXDYAl8mfaaemZzd7zyk0w5A3-LQ/view?usp=sharing) and push the file at `runs/TRY_FOLDER`
## Train
Execute `run.ipynb` in the folder

## Explanation
### Dataset
[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) was used as used in original paper

### Augmentation
- `crop-and-resize` with random horizontal flip, images are resized to 32x32
- `color distorition` that code provided in the original paper (Appendix A)
- `random grayscale`
- `Gaussian blur` using kernel size is 32

### Encoder
`ResNet18`, `ResNet50` were used as backbone network, for details see `model/resnet.py` <br>
(Remove the final fully connected layer, giving a representation dimension 128)

### Projected head
Add MLP projection head (Fully Connected layer) to backbone network resnet
- 2 hidden layer of projected head reduce dimensions
  - `ResNet18`: 512 -> 512 -> 128
  - `ResNet50`: 2048 -> 2048 -> 128
- with `ReLU` activation function

### Loss
Introducing a learnable nonliner transformation between the representation and the contrastive loss substantially imporoves the quality of the learned representations
 - `InfoNCE` loss was used to contrastive learning
 - The similarity was calculated by cosine similarity
 - The loss scores are calculated by `CrossEntropyLoss` with normailzed temperature-scaled logits

### Opimmizer
`Adam` was used with `CosineAnnealingLR` scheduler

### Result
Performance under different batch sizes using modified ResNet50, epochs = 400, t = 0.07, lr = 0.0003
| source | 256 | 512 |
| ------ | ------| ------|
| paper | 83.9 % | 84.2 % |
| this repo | 74.02 % | 78.32 % |

### Outstanding difference with the original paper
- original paper was used `NT-Xent` loss, but this code used `InfoNCE`
- `Adam` optimizer was used instead of `LARS`

## TODO
- [ ] Train another dataset

## Reference
[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)<br>
[PyTorch implementation of SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://github.com/sthalles/SimCLR)
