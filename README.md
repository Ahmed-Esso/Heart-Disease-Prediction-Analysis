# ❤️ Heart Disease Prediction Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-prediction-analysis-ahmed-essam.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive machine learning project for predicting heart disease using the UCI Heart Disease Dataset. This project combines data analysis, visualization, and multiple machine learning algorithms to create an accurate heart disease prediction system with an interactive web interface.

## 🌟 Features

- **📊 Comprehensive Data Analysis**: In-depth exploratory data analysis with statistical insights
- **🎯 Multiple ML Models**: Comparison of Random Forest, Logistic Regression, SVM, and KNN algorithms
- **🚀 Interactive Web App**: Real-time predictions through a user-friendly Streamlit interface
- **📈 Rich Visualizations**: Interactive charts and graphs using Plotly
- **🔍 Feature Analysis**: Correlation analysis and feature importance visualization
- **⚡ Real-time Predictions**: Instant heart disease risk assessment

## 🎯 Live Demo

Try the live application: [Heart Disease Prediction App](https://heart-disease-prediction-analysis-ahmed-essam.streamlit.app/)

## 📸 Screenshots

### Dashboard Overview
<img width="1920" height="1080" alt="Screenshot 2025-08-28 003201" src="https://github.com/user-attachments/assets/47713bc6-e93b-473d-a5d5-43301ab23bf4" />

### Model Comparison
<img width="1920" height="1080" alt="Screenshot 2025-08-28 003144" src="https://github.com/user-attachments/assets/b5fe19d0-ec25-46d0-b54d-fa14eab82bf3" />

### Individual Prediction
<img width="1920" height="1080" alt="Screenshot 2025-08-28 003129" src="https://github.com/user-attachments/assets/62fa2110-48b2-45fd-baca-cf4ee72a757c" />

## 🏗️ Project Structure

```
Heart-Disease-Prediction-Analysis/
├── 📁 data/
│   └── processed.cleveland.data    # UCI Heart Disease Dataset
├── 📁 notebooks/
│   └── Heart.ipynb               # Jupyter notebook with full analysis
├── 📁 src/
│   └── streamlit_app.py         # Main Streamlit application
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                # Project documentation
└── 📄 LICENSE                  # License file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Heart-Disease-Prediction-Analysis.git
   cd Heart-Disease-Prediction-Analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv heart_disease_env
   
   # On Windows
   heart_disease_env\Scripts\activate
   
   # On macOS/Linux
   source heart_disease_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 🏃‍♂️ Running the Application

#### Streamlit Web App
```bash
streamlit run src/streamlit_app.py
```
Then navigate to `http://localhost:8501` in your browser.

#### Jupyter Notebook Analysis
```bash
jupyter notebook notebooks/Heart.ipynb
```

## 📊 Dataset Information

The project uses the **UCI Heart Disease Dataset** with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Numeric |
| `sex` | Gender (1 = male, 0 = female) | Categorical |
| `cp` | Chest pain type (0-3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results (0-2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numeric |
| `exang` | Exercise induced angina | Binary |
| `oldpeak` | ST depression induced by exercise | Numeric |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels (0-3) | Numeric |
| `thal` | Thalassemia (0-3) | Categorical |
| `target` | Heart disease diagnosis | Binary |

**Dataset Statistics:**
- 📈 Total samples: 303
- 🎯 Features: 13
- 📊 Target classes: Binary (Disease/No Disease)
- ✅ Preprocessing: Missing values handled

## 🤖 Machine Learning Models

### Model Performance Comparison

| Model | Accuracy | Strengths |
|-------|----------|-----------|
| **🔍 K-Nearest Neighbors** | **91.8%** | Simple, effective for local patterns, excellent performance |
| 🎯 Support Vector Machine | 90.2% | Good with non-linear data, robust |
| 🌳 Random Forest | 86.9% | Handles non-linear relationships, good interpretability |
| 📈 Logistic Regression | 85.5% | High interpretability, fast training |

### Key Insights

- **🏆 Best Model**: K-Nearest Neighbors with 91.8% accuracy
- **🔑 Top Predictive Features**: 
  1. Maximum heart rate (`thalach`)
  2. Chest pain type (`cp`)
  3. Thalassemia (`thal`)
  4. Exercise induced angina (`exang`)
  5. ST depression (`oldpeak`)

## 📱 Application Features

### 🏠 Overview Page
- Dataset statistics and summary
- Missing value analysis
- Data quality indicators

### 🔍 Data Exploration
- Target variable distribution
- Feature statistics
- Data type information

### 📊 Visualizations
- Age distribution analysis
- Gender-based disease patterns
- Correlation heatmaps
- Feature importance plots

### 🤖 Model Comparison
- Algorithm performance metrics
- Cross-validation results
- Model selection rationale

### 🔮 Individual Prediction
- Interactive input form
- Real-time risk assessment
- Probability visualization
- Clinical decision support

## 🛠️ Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
plotly>=5.15.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 🔬 Model Development Process

1. **Data Preprocessing**
   - Missing value imputation
   - Feature scaling for specific algorithms
   - Target variable encoding

2. **Exploratory Data Analysis**
   - Statistical analysis
   - Correlation studies
   - Distribution analysis

3. **Feature Engineering**
   - Feature selection based on correlation
   - Importance ranking
   - Domain knowledge integration

4. **Model Training & Evaluation**
   - Cross-validation
   - Hyperparameter tuning
   - Performance comparison

5. **Model Deployment**
   - Streamlit web interface
   - Real-time prediction system

## 🚀 Deployment

The application is deployed on Streamlit Cloud and automatically updates with repository changes.

### Local Deployment
```bash
streamlit run src/streamlit_app.py --server.port 8501
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the heart disease dataset
- **Streamlit** for the amazing web app framework
- **Plotly** for interactive visualizations
- **scikit-learn** for machine learning algorithms

## 📞 Contact

**Ahmed Essam** - [GitHub Profile](https://github.com/yourusername)

Project Link: [https://github.com/yourusername/Heart-Disease-Prediction-Analysis](https://github.com/yourusername/Heart-Disease-Prediction-Analysis)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

## 🔮 Future Enhancements

- [ ] Integration with more advanced ML models (XGBoost, Neural Networks)
- [ ] API endpoint for predictions
- [ ] Mobile-responsive design improvements
- [ ] Integration with medical databases
- [ ] Multi-language support
- [ ] Advanced feature engineering
- [ ] Model interpretability with SHAP values

---

*This project is for educational and research purposes. Always consult healthcare professionals for medical decisions.*
