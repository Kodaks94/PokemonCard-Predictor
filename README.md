# Pokémon Card Predictor  

A deep learning pipeline for classifying Pokémon TCG cards using image-based recognition, MobileNetV2, and LSTM memory.

## Overview

This project builds a full image classification pipeline for Pokémon trading cards:
-  Downloads card images from a CSV
-  Generates structured labels
-  Applies data augmentation (especially for rare/singleton cards)
-  Trains a MobileNetV2 + BiLSTM classifier
-  Evaluates and plots performance metrics
-  Saves models, metrics, and predictions for reproducibility

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## ️ Features
- **MobileNetV2** base for efficient feature extraction
- **BiLSTM head** to capture sequential/spatial dependencies
- **Mixed Precision** training enabled for speed on RTX 4090+
- **Data augmentation** for underrepresented classes
- **Auto-labeling** and dynamic dataset generation
- Compatible with TensorFlow 2.10+ with GPU support

##  Project Structure

```
PokemonCard-Predictor/
│
├── pokemoncards/
│   ├── pokemon-cards.csv                # Metadata with image URLs
│   ├── TCG_labels                       # Generated label file
│   └── pokemon_card_images/            # Downloaded card images
│       └── augmented/                  # Augmented images
│
├── CNN-TCG-Model.py                     # Full model training pipeline
├── tcg_download_images.py              # Downloads card images from CSV
├── tcg_label_generator.py              # Creates label index CSV
├── models/                              # Saved models & reports
│
├── .gitignore
└── README.md
```

## Data Source

This project uses the **Pokémon TCG** dataset sourced from [Kaggle - Pokémon Cards by Priyam Choksi](https://www.kaggle.com/datasets/priyamchoksi/pokemon-cards), which contains metadata and image URLs for a wide range of Pokémon trading cards.

## Installation

```bash
conda create -n pokemon python=3.8
conda activate pokemon

# Install dependencies
pip install -r requirements.txt

# Or manually:
pip install tensorflow==2.10
pip install pandas scikit-learn matplotlib tqdm requests
```

**GPU support:** Make sure your TensorFlow is compiled with CUDA 11.2 + cuDNN 8.6. Confirm with:
or you can ignore this by training on your CPU
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

##  Usage Guide

### 1. Download Card Images

```bash
python tcg_download_images.py
```

This script will:
- Download images using URLs in `pokemon-cards.csv`
- Save images to `pokemoncards/pokemon_card_images`
- Apply **data augmentation** for each downloaded image and store them in `augmented/`

### 2. Generate Label File

```bash
python tcg_label_generator.py
```

Creates a CSV mapping each image path to its label name and label index → saved as `pokemoncards/TCG_labels`.

### 3. Train the Classifier

```bash
python CNN-TCG-Model.py
```

This:
- Loads all real + augmented data
- Trains a MobileNetV2 + BiLSTM classifier
- Uses early stopping and checkpoint saving
- Outputs:
  - `models/accuracy_plot.png` and `loss_plot.png`
  - `models/classification_report.txt`
  - Trained model in `.keras` format

## Example Results (Sample Output)

| Metric       | Value         |
|--------------|---------------|
| Top Accuracy | 83.5%         |
| Validation Accuracy | ~20% (real-world test scenario) |
| GPU Runtime | ~14s per epoch (RTX 4090) |

##  Possible Improvements
-  Use CLIP-style embeddings for semantic grouping
-  Turn into a gradio/streamlit demo
-  Add OCR for card text interpretation
-  Balance class distributions

## .gitignore

```
.venv/
__pycache__/
pokemoncards/pokemon_card_images/
models/
*.keras
```

## 🧑‍💻 Author

Made with pain, GPU tears, and persistence by [@Kodaks94](https://github.com/Kodaks94)
