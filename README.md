# Brain Cancer MRI Classification Project

## Project goal
The aim of this project is to develop a reliable and efficient image classification model that uses Convolutional Neural Networks (CNNs) to analyze medical image data and classify brain tumors. This should support the diagnostic processes, improve the accuracy of the results and reduce the workload of radiologists.

## Motivation
Brain tumors are one of the most serious diseases that require early and accurate diagnosis to improve the prognosis of patients. Traditional diagnostic techniques are based on manual analysis of MRI or CT images, which can be time-consuming and error-prone. The use of CNNs in image processing offers the potential to detect subtle patterns in medical images that are difficult for the human eye to identify.

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

```
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


