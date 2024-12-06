# Brain Cancer MRI Classification Project

## Overview
This project focuses on image classification of brain cancer MRI scans using machine learning and deep learning techniques. The project utilizes the Bangladesh Brain Cancer MRI Dataset to develop and train models for automated medical image diagnostics.

## Dataset Information
- **Dataset Name**: Bangladesh Brain Cancer MRI Dataset
- **Publication Date**: 5 August 2024
- **Version**: 1
- **DOI**: 10.17632/mk56jw9rns.1
- **Contributor**: Md Mizanur Rahman
- **Institution**: Daffodil International University

### Dataset Composition
- Total Images: 6,056
- Image Classes:
  - Brain Glioma: 2,004 images
  - Brain Menin: 2,004 images
  - Brain Tumor: 2,048 images

### Dataset Characteristics
- Image Size: 512x512 pixels
- Source: Various hospitals across Bangladesh
- Collected with medical professional involvement

## Project Goals
- Develop an accurate image classification model for brain cancer MRI scans
- Assist in early diagnosis and medical image interpretation
- Demonstrate the application of machine learning in medical diagnostics


## Project Structure
```
brain-cancer-mri-classification/
│
├── data/
|   ├── CSV                 # data in csv structure
|   ├── Images/             # images
|          ├── Glioma
|          ├── Menin
|          ├── Tumor            
├── data exploration/       # Jupyter notebook for exploration
|   ├── plots               # plots of the distributions
├── results/                
│   ├── plots               # plots of metrics
├── session/                # contains saved models
├── args.py
├── dataset.py
├── evaluate.py
├── helper.py
├── main.py                 # main file to run pipeline
├── model.py                # Model architecture
├── net.py                  # custom net
├── trainer.py              # Training script
└── README.md               # Project documentation


## Evaluation Metrics
- Accuracy
- Precision
- Recall
- Confusion Matrix

## Limitations
- Dataset is specific to Bangladeshi hospital data
- Limited sample size
- Potential bias in image collection

## License 
This project is under MIT license
## Citation
If you use this dataset or project in your research, please cite:
```
Rahman, Md Mizanur (2024). Bangladesh Brain Cancer MRI Dataset. 
DOI: 10.17632/mk56jw9rns.1
```
