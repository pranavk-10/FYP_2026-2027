# HAM10000 Skin Lesion Classification

A Final Year Project (FYP) for classifying skin lesions using deep learning. This project uses the HAM10000 dataset to build a Convolutional Neural Network (CNN) model for skin cancer detection.

## Project Overview

This project aims to classify skin lesions into diagnostic categories using PyTorch-based deep learning models. The HAM10000 dataset contains dermatological images with associated metadata including patient demographics and diagnostic classifications.

## Dataset

- **Dataset Name**: HAM10000 (Human Against Machine with 10000 training images)
- **Location**: `Skin Dataset/`
- **Metadata**: `Skin Dataset/metadata/HAM10000_metadata.csv`
- **Records**: 10,000 skin lesion images with diagnostic labels
- **Features**:
  - `lesion_id`: Unique identifier for the lesion
  - `image_id`: Corresponding image ID
  - `dx`: Diagnostic classification (e.g., bkl = benign keratosis-like)
  - `dx_type`: Type of diagnosis (histo = histopathology confirmed)
  - `age`: Patient age
  - `sex`: Patient sex
  - `localization`: Body location of the lesion
  - `dataset`: Source dataset

## Project Structure

```
FYP_2026-2027/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── notebooks/
│   └── 01_eda.ipynb                   # Exploratory Data Analysis
└── Skin Dataset/
    ├── images/                         # Skin lesion images
    └── metadata/
        └── HAM10000_metadata.csv       # Metadata and labels
```

## Dependencies

The project uses the following Python libraries:

- **Deep Learning**: PyTorch, Torchvision
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Image Processing**: Pillow, OpenCV
- **Visualization**: Matplotlib, Seaborn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Getting Started

1. Install the required dependencies from `requirements.txt`
2. Run the Exploratory Data Analysis notebook: `notebooks/01_eda.ipynb`
3. Develop and train models for skin lesion classification

## License

Project created for FYP 2026-2027
