# OLTC-Monitoring-Audio
This repository contains the code, data and additional materials of the paper "Transformer OLTC operation monitoring framework through Accoustic Signal Processing and Convolutional Neural Neworks" by Adnan Secic (DV Power, Sweden), Jose I. Aizpurua (University of the Basque Country and Ikerbasque Basque Foundation for Science, Spain), Unai Garro (Mondragon University, Spain), Eñaut Muxika (Mondragon University, Spain) and Igor Kuzle (University of Zagreb, Croatia) published in IEEE Transactions on Instrumentation and Measurement 

# Elin State Tagging – Deep Neural Network Training & Testing

MATLAB implementation for training and testing deep neural networks on **elin state tagging** data.  
Supports undersampling strategies, STFT-based feature extraction, mislabeled data correction, and iterative dataset generation.

---

## Quick Start

### Training
Run:
```matlab
elin_state_tagging_script.m
```
- Choose undersampling strategy (3 options).  
- Set number of iterations (STFT step = 1024 / input).  
- Select training/validation ratio (default 0.5).  
- Train new model or validate existing one.  
- Optionally correct mislabeled data.  

💡 Example: Input `4` → step = 256 samples → ~1h training.

---

### Testing
Run:
```matlab
elin_state_tagging_test_script.m
```
- Choose number of networks (default 3).  
- Select trained networks from folder.  
- Set iterations (same STFT method as training).  
  - ⚠️ Input `1024` → testing with 1-sample step size → very slow.  
- Training/validation ratio (default 0.5).  
- Optionally correct mislabeled data.  

---

## Requirements
- MATLAB R2021a or later  
- Deep Learning Toolbox  
- Signal Processing Toolbox  

---

## License
MIT License – feel free to use and modify.
