# BERT Fine-tuning on 20 Newsgroups

## Overview
Fine-tune a BERT model on the 20 Newsgroups dataset for text classification 

## Dataset
- **Source:** SetFit/20_newsgroups from HuggingFace
- **Training samples:** 11,314
- **Test samples:** 7,532
- **Classes:** 20 newsgroup categories

## Model
- **Base Model:** BERT (bert-base-uncased)
- **Task:** Multi-class text classification
- **Framework:** PyTorch 

## Results

### Evaluation Metrics (Test Set)
- **Accuracy:** 71.44%
- **Precision (weighted):** 71.53%
- **Recall (weighted):** 71.44%
- **F1-Score (weighted):** 71.15%

### Training Metrics
- **Final Training Accuracy:** 84.52%
- **Final Training Loss:** 0.5403

## Files
- `eda.py` - Exploratory Data Analysis with class distribution visualization
- `train.py` - BERT fine-tuning using PyTorch training loop
- `test.py` - Evaluation framework with pretrained BERT
- `Main.py ` - Custom data to test the classification

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Exploratory Data Analysis
```bash
python eda.py
```
Outputs: `class_distribution.png`

### 2. Train Model
```bash
python train.py
```
Outputs: Fine-tuned model saved to `./bert-20newsgroups/`
Time: ~90 minutes on GPU

### 3. Evaluate & Test
```bash
python test.py
```
Outputs: Metrics, `confusion_matrix.png`, and inference examples

### 4. Custom Predictions
```bash
python mian.py
```
Test model on custom text examples

## Key Implementation Details

### Training Loop (PyTorch Only)
- Learning rate: 2e-5 with linear schedule and warmup
- weight decay: 0.01 
- Batch size: 16
- Epochs: 2 seeds 4 and 3 

### Data Preprocessing
- Tokenization using BertTokenizer
- Max sequence length: 512 tokens
- Padding and truncation applied
- Train/test split: 60/40 (provided by dataset)

### Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score (all weighted)
- Confusion Matrix: 20x20 heatmap visualization
- Inference: predict_text() function returns class label and confidence score

## Performance Notes
- Training accuracy (84.52%) higher than test accuracy (71.44%) indicates some overfitting
- Model generalizes well to new text examples (5/6 custom examples correct)
- Classes with distinct vocabulary (e.g., religion, science) have higher prediction accuracy

## Requirements
See `requirements.txt` for all dependencies
