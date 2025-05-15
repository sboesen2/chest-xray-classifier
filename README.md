# Chest X-ray Disease Classifier

A Streamlit-based web application for classifying chest X-ray images into 15 different disease categories using deep learning.

## Overview

This project implements a multi-label classification system for chest X-rays using DenseNet121 architecture. The application provides:
- Disease probability predictions
- Uncertainty estimation using MC Dropout
- Grad-CAM visualization for model interpretability
- Detailed model performance metrics and comparisons

## Models

The application includes two models:
1. `best_densenet.pt` (Final Model)
   - Strict patient-level split evaluation
   - More realistic performance metrics
   - Better generalization to unseen patients

2. `densenet121_xray.pt` (Initial Model)
   - Higher but potentially overestimated performance
   - Used for comparison purposes

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv xray
xray\Scripts\activate  # Windows
source xray/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your model files in the `models/` directory:
   - `models/best_densenet.pt`
   - `models/densenet121_xray.pt`

## Running the Application

```bash
streamlit run Home.py
```

The application will be available at `http://localhost:8501`

## Features

- Upload and analyze chest X-ray images
- Get probability scores for 15 different conditions
- View uncertainty estimates for predictions
- Generate Grad-CAM visualizations
- Compare model performance metrics
- Detailed model information and methodology

## Model Performance

- Final Model (best_densenet.pt):
  - Overall AUC: 0.5499
  - Best performing class: Effusion (AUC = 0.6903)
  - Strict patient-level split evaluation

- Initial Model (densenet121_xray.pt):
  - Overall AUC: 0.7657
  - Best performing class: Cardiomegaly (AUC = 0.8760)
  - Used for comparison purposes

## Acknowledgments

This project was made possible through the use of Arizona State University's Sol HPC supercomputer. The computational resources provided by ASU's Research Computing were essential for training and evaluating the deep learning models used in this application. Special thanks to the ASU Research Computing team for their support and infrastructure.

## Created by

Sam Boesen

## License

Idgaf about no license open source biology is king

## Disclaimer

This application is for research and educational purposes only. Always consult healthcare professionals for medical decisions. 