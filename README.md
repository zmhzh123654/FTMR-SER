# FTMR-SER
Code for Frequency-Temporal Multi-Representation Learning Combines Feature Alignment and Self-Distillation for Speech Emotion Recognition
# File system
```
├── FTMR-SER
│   ├── models
│   │   └── transformers_encoder
│   │       ├── multihead_attention.py    # Multi-head attention module
│   │       ├── position_embedding.py     # Positional embedding
│   │       └── transformer.py            # Transformer block
│   ├── ser_model.py                      # SER model definition
│   ├── data_utils.py                     # Data loading and preprocessing
│   └── train.py                          # Training script
├── README.md
└── requirements.txt                       # Package dependencies
```
# Environmet
PyTorch 2.1.2
Python 3.10
CUDA 11.8
NVIDIA RTX 4090 GPU (24 GB memory)
