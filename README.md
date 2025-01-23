# Iris Flower Classification Web Application

## Project Overview
This is an interactive web application for classifying Iris flowers based on their measurements. The project uses machine learning to predict the species of Iris flowers (setosa, versicolor, or virginica) using sepal and petal measurements.

## Features
- Interactive web interface built with Streamlit
- Real-time prediction of Iris flower species
- Support for multiple machine learning models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Standardized data processing
- Probability scores for predictions
- User-friendly input validation

## Installation

### Prerequisites
- Python 3.x
- pip (Python package manager)

### Setup Instructions
1. Clone this repository:
```bash
cd project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter the following measurements in the input fields:
   - Sepal Length (cm)
   - Sepal Width (cm)
   - Petal Length (cm)
   - Petal Width (cm)

4. Click the "Predict Species" button to see the classification results

## Project Structure
```
project/
├── app.py                 # Main Streamlit web application
├── model_selection.py     # Model training and selection script
├── dataset/              # Contains the Iris dataset
├── model/               # Saved model and scaler files
└── requirements.txt      # Project dependencies
```

## Model Training
The project includes a comprehensive model selection process (`model_selection.py`) that:
1. Loads and preprocesses the Iris dataset
2. Scales features using StandardScaler
3. Trains multiple classification models
4. Evaluates models using cross-validation
5. Saves the best performing model for use in the web application

## Technical Details
- **Framework**: Streamlit
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib

## Input Requirements
- All measurements should be in centimeters (cm)
- Valid range: 0.0 to 10.0 cm
- Step size: 0.1 cm

## Output
The application provides:
- Predicted Iris species
- Confidence scores for each possible species
- Visual representation of the prediction probabilities

## Error Handling
The application includes robust error handling for:
- Model loading issues
- Invalid input validation
- Runtime prediction errors



