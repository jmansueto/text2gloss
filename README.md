# Text2Gloss: Transformer-based Sign Language Translation

This repository implements transformer models for translating text into sign language gloss annotations, providing a baseline for end-to-end sign language translation and supporting sign language interpreters. The project explores various approaches including multilingual training and POS-tagging augmentation to improve performance on low-resource sign language translation tasks.

## Datasets

The project supports two main datasets:

1. **ASLG-PC12**: American Sign Language Gloss Parallel Corpus 2012
2. **PHOENIX-14T**: German Sign Language dataset from PHOENIX-2014T

## Model Architectures

The repository includes three different transformer model implementations:

1. **Basic Transformer** (`basic_transformer/`): Standard transformer architecture
2. **Transformer Tiny** (`transformer_tiny/`): Smaller transformer variant for faster training
3. **Transformer from Pretrained XLM** (`transformer_from_pretrained_xlm/`): Leverages pretrained cross-lingual language models

## Repository Structure

```
text2gloss/
├── code/
│   ├── eval/
│   │   └── evaluate.py              # BLEU score evaluation
│   ├── models/
│   │   ├── basic_transformer/       # Standard transformer model
│   │   ├── transformer_tiny/        # Compact transformer model
│   │   ├── transformer_from_pretrained_xlm/  # Pretrained XLM model
│   │   └── logs/                    # Training logs and TensorBoard files
│   └── preprocessing/
│       ├── load_aslg_pc12.py        # ASLG-PC12 dataset loader
│       ├── load_phoenix.py          # PHOENIX dataset loader
│       ├── preprocess.py            # Main preprocessing pipeline
│       ├── src_pos_splits.py        # POS-tagging augmentation
│       └── standard_splits.py       # Standard train/val/test splits
├── environment.yml                  # Conda environment specification
└── README.md
```

## Setup

### Prerequisites

- Python 3.8
- CUDA-compatible GPU (recommended)
- Conda package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd text2gloss
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate text2gloss
   ```

3. **Install additional dependencies:**
   The environment includes all necessary packages including:
   - PyTorch 2.2.1
   - Fairseq 0.12.2
   - Transformers and other ML libraries
   - Data processing tools (pandas, scikit-learn)
   - Evaluation metrics (sacrebleu)

### Dataset Setup

#### ASLG-PC12 Dataset
The ASLG-PC12 dataset is automatically downloaded from Hugging Face datasets:

```bash
cd code/preprocessing
python load_aslg_pc12.py
```

#### PHOENIX-14T Dataset
For the PHOENIX dataset, you need to download it from Kaggle:

1. Set up Kaggle credentials:
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

2. Download and process the dataset:
   ```bash
   kaggle datasets download -d mariusschmidtmengin/phoenixweather2014t-3rd-attempt
   python load_phoenix.py
   ```

## Usage

### Data Preprocessing

1. **Standard preprocessing:**
   ```bash
   cd code/preprocessing
   python preprocess.py --task translation --tokenizer moses --source_lang en --target_lang asl --workers 4 --bpe_type subword_nmt
   ```

2. **POS-tagging augmentation:**
   ```bash
   python src_pos_splits.py  # Adds POS tags to source text
   ```

### Model Training

#### Basic Transformer
```bash
cd code/models/basic_transformer
chmod +x train.sh
./train.sh
```

#### Transformer Tiny
```bash
cd code/models/transformer_tiny
chmod +x train.sh
./train.sh
```

#### Pretrained XLM Transformer
```bash
cd code/models/transformer_from_pretrained_xlm
chmod +x train.sh
./train.sh
```

### Model Evaluation

1. **Generate translations:**
   ```bash
   cd code/models/basic_transformer
   chmod +x test.sh
   ./test.sh
   ```

2. **Evaluate with BLEU score:**
   ```bash
   cd code/eval
   python evaluate.py
   ```

### Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir code/models/logs/
```

## Key Features

- **Multiple Model Architectures**: Standard, tiny, and pretrained transformer variants
- **Data Augmentation**: POS-tagging and multilingual training support
- **Comprehensive Evaluation**: BLEU score evaluation with sacrebleu
- **Flexible Preprocessing**: Support for different tokenizers and BPE encoding
- **Low-Resource Optimization**: Specialized approaches for sign language translation

## Configuration

### Training Parameters

- **Epochs**: 30 (configurable in training scripts)
- **Learning Rate**: 5e-4
- **Batch Size**: Dynamic based on max tokens (2048)
- **Optimizer**: Adam with inverse square root scheduler
- **Warmup Steps**: 4000
- **Dropout**: 0.1

### Data Processing

- **Tokenization**: Moses tokenizer
- **BPE**: Subword-nmt with 32,000 merge operations
- **Train/Val/Test Split**: 80%/10%/10%

## Results

The models are evaluated on ASLG-PC12 and PHOENIX-14T datasets, achieving improved performance relative to baselines with multilingual corpus training and data augmentation techniques.


