# Heart Disease Prediction Project

A comprehensive machine learning project for predicting heart disease using the UCI heart disease dataset.

## Features

- **Data Analysis**: Comprehensive exploratory data analysis with visualizations
- **Machine Learning**: Multiple ML models comparison (Random Forest, Logistic Regression, SVM, KNN)
- **Interactive Web App**: Streamlit application for real-time predictions
- **Jupyter Notebook**: Complete analysis workflow
live app https://heart-disease-prediction-analysis-ahmed-essam.streamlit.app/
## Project Structure

```
├── data/                   # Dataset files
├── notebooks/             # Jupyter notebooks
│   └── Heart.ipynb       # Main analysis notebook
├── src/                   # Source code
│   └── streamlit_app.py  # Streamlit web application
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Jupyter Notebook Analysis
Open and run `notebooks/Heart.ipynb` for complete data analysis.

### Streamlit Web Application
Run the interactive web app:
```bash
streamlit run src/streamlit_app.py
```

## Dataset

The project uses the UCI Heart Disease dataset with the following features:
- Age, sex, chest pain type
- Blood pressure, cholesterol levels
- Heart rate, exercise-induced angina
- And more clinical indicators

## Models

The project compares multiple machine learning models:
- **Random Forest**: Best performing model (~88% accuracy)
- **Logistic Regression**: Good interpretability
- **Support Vector Machine**: Non-linear classification
- **K-Nearest Neighbors**: Instance-based learning

## Results

- Random Forest achieved the highest accuracy (88%)
- Key predictive features: maximum heart rate, chest pain type, thalassemia
- Model suitable for clinical decision support

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- plotly

## License

This project is for educational purposes.
