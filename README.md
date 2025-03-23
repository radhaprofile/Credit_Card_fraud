# Credit Card Fraud Detection - README

## Project Overview
This project aims to detect fraudulent credit card transactions using machine learning. A Random Forest Classifier is trained on a dataset containing transactions from European cardholders in September 2013.

## Folder Structure
```

├── creadit_card_fraud.py      # Python script for model training
├── credit_fraud_model.pkl  # Trained machine learning model
├── report.pdf              # Project report
├── README.md               # Documentation
```

## Installation & Setup
### 1. Clone the Repository
```
git clone https://github.com/radhaprofile/Credit_Card_fraud.git
cd Credit_Card_fraud
```

### 2. Install Dependencies
Make sure you have Python 3.x installed. Then, install required libraries:
```
pip install -r requirements.txt
```

### 3. Run the Model Training Script
```
python creadit_card_fraud.py
```

## Model Details
- **Algorithm Used:** Random Forest Classifier
- **Handling Imbalanced Data:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Performance Metrics:** Accuracy, Precision, Recall, Confusion Matrix
- **Final Model Accuracy:** >75%

## Deployment Plan
The model can be deployed using Flask or FastAPI for real-time fraud detection.

## Future Enhancements
- Implement deep learning models for improved performance.
- Integrate with a real-time fraud detection API.
- Optimize hyperparameters further.

## Contact
For any questions, please reach out at [your_email@example.com].

