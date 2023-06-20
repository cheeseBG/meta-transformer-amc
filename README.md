# 

## Toward Scalable Automatic Modulation Classification via Meta-Learning
This paper is under review.

>**Abstract:** *Driven by recent technological breakthroughs in deep learning (DL), many recent automatic modulation classification (AMC) methods utilize deep networks to classify the type of modulation in the incoming signal at the receiver. However, existing DL-based approaches suffer from limited scalability, especially for unseen modulations or input signals from new environments not used in training the DL model, thus not ready for real-world systems such as software defined radio devices. 
In this paper, we introduce a scalable AMC scheme that provides flexibility for new modulations and adaptability to input signals with diverse configurations. We propose a meta-learning framework based on few-shot learning (FSL) to acquire general knowledge and a learning method for AMC tasks. This approach allows the model to recognize new unseen modulations by learning
with only a very small number of samples, without requiring the entire model to be retrained. Additionally, we enhance the scalability of the classifier by leveraging a transformer-based encoder, enabling efficient processing of input signals with varying configurations. Extensive evaluations demonstrate that the proposed AMC method outperforms existing techniques across all signal-to-noise ratios (SNRs) on RadioML2018.01A dataset.*

## Computing environment
This project has been developed and tested under the following computing environment:
- Ubuntu 20.04
- Intel(R) i9-9900KF
- GeForce RTX 2080 Ti 11GB
- Python 3.8

## Dataset and pre-trained model
1. Click [here](https://www.deepsig.ai/datasets) to download the RadioML2018.01A dataset.

2. Click [google drive](https://drive.google.com/file/d/1x_amkYkb3m3bdpPeRztMWhJUWxmBSzmk/view?usp=sharing) to download pre-trained models

2. Place the downloaded dataset and pre-trained models as follows:
```
meta-transformer-amc
   |——————amc_dataset
   |        └——————RML2018
   |                └——————classes-fixed.json
   |                └——————GOLD_XYZ_OSC.0001_1024.hdf5
   └——————checkpoint
            └——————learning
            └——————size
            └——————unseen
 
```


## Installation

* Clone the repository and install the required dependencies
```
pip install -r requirements.txt
```
## Usage

```
cd AMC

python main.py fs
```





## Overview of meta-learning architecture 
<div align="center">
<img src="./paper_figures/figures/overview.jpg" width="780" height="450"/>
</div>

## Few-shot learning evaluation
<div align="center">
<img src="./paper_figures/figures/fewshot.png" width="400" height="300"/> 
</div>

## Input size scalability evaluation
<div align="center">
<img src="./paper_figures/figures/size_cnn.png" width="400" height="250"/>
<img src="./paper_figures/figures/size_vit.png" width="400" height="250"/>
</div>



