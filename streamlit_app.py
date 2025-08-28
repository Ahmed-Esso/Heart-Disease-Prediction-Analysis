import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_heart_data():
    """Load and preprocess the heart disease dataset"""
    try:
        # Column names for the dataset
        col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        # Try to load the data from the expected location
        data_path = 'processed.cleveland.data'
        df = pd.read_csv(data_path, names=col, na_values='?')
        
        # Handle missing values
        df['ca'].fillna(df['ca'].mode()[0], inplace=True)
        df['thal'].fillna(df['thal'].mode()[0], inplace=True)
        
        # Create binary target (0=no disease, 1=disease)
        df['target_binary'] = (df['target'] > 0).astype(int)
        
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure the heart disease dataset is available.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def train_models(df):
    """Train multiple ML models and return results"""
    # Prepare features and target
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    X = df[feature_cols]
    y = df['target_binary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # Use scaled data for some models
        if name in ['Logistic Regression', 'SVM', 'KNN']:
            X_train_use, X_test_use = X_train_scaled, X_test_scaled
        else:
            X_train_use, X_test_use = X_train, X_test
        
        # Train model
        model.fit(X_train_use, y_train)
        y_pred = model.predict(X_test_use)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({'Model': name, 'Accuracy': accuracy})
        trained_models[name] = {
            'model': model,
            'scaler': scaler if name in ['Logistic Regression', 'SVM', 'KNN'] else None,
            'features': feature_cols
        }
    
    return results, trained_models, X_test, y_test

def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
    
    st.title("❤️ Heart Disease Prediction Analysis")
    st.markdown("*Interactive analysis of the UCI heart disease dataset*")
    
    # Load data
    df = load_heart_data()
    
    if df is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Overview", "Data Exploration", "Visualizations", 
                                "Model Comparison", "Individual Prediction"])
    
    if page == "Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 2)  # Excluding target columns
        with col3:
            st.metric("Disease Cases", sum(df['target_binary'] == 1))
        with col4:
            st.metric("Healthy Cases", sum(df['target_binary'] == 0))
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            st.success("✅ No missing values in the dataset!")
        else:
            st.dataframe(missing_values[missing_values > 0])
    
    elif page == "Data Exploration":
        st.header("🔍 Data Exploration")
        
        # Target distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Heart Disease Distribution")
            target_counts = df['target_binary'].value_counts()
            fig = px.pie(values=target_counts.values, 
                        names=['No Disease', 'Disease'],
                        color_discrete_sequence=['lightcoral', 'lightgreen'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Disease Levels (Original)")
            original_counts = df['target'].value_counts().sort_index()
            fig = px.bar(x=original_counts.index, y=original_counts.values,
                        labels={'x': 'Disease Level', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature statistics
        st.subheader("Feature Statistics")
        st.dataframe(df.describe())
    
    elif page == "Visualizations":
        st.header("📈 Data Visualizations")
        
        # Age distribution
        st.subheader("Age Distribution by Disease Status")
        
        no_disease = df[df['target_binary'] == 0]['age']
        disease = df[df['target_binary'] == 1]['age']
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=no_disease, name='No Disease', 
                                 opacity=0.7, marker_color='lightcoral', nbinsx=15))
        fig.add_trace(go.Histogram(x=disease, name='Disease', 
                                 opacity=0.7, marker_color='lightgreen', nbinsx=15))
        
        fig.update_layout(
            title="Age Distribution by Disease Status",
            xaxis_title="Age",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gender vs Disease
        st.subheader("Gender Distribution")
        gender_disease = pd.crosstab(df['sex'], df['target_binary'])
        gender_disease.index = ['Female', 'Male']
        gender_disease.columns = ['No Disease', 'Disease']
        
        fig = px.bar(gender_disease, barmode='group',
                    title="Heart Disease by Gender")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target_binary']
        corr_matrix = df[features].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       color_continuous_midpoint=0,
                       title="Feature Correlation Heatmap")
        fig.update_layout(
            width=800,
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations with target
        st.subheader("Top Correlations with Heart Disease")
        target_corr = corr_matrix['target_binary'].abs().sort_values(ascending=False)
        top_corr = target_corr.drop('target_binary').head(5)
        
        fig = px.bar(x=top_corr.values, y=top_corr.index, orientation='h',
                    title="Features Most Correlated with Heart Disease")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Comparison":
        st.header("🤖 Machine Learning Model Comparison")
        
        with st.spinner("Training models..."):
            results, trained_models, X_test, y_test = train_models(df)
        
        # Model accuracy comparison
        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
        
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df, x='Accuracy', y='Model', orientation='h',
                        title="Model Accuracy Comparison")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(results_df.style.format({'Accuracy': '{:.3f}'}))
        
        # Best model details
        best_model_name = results_df.iloc[0]['Model']
        best_accuracy = results_df.iloc[0]['Accuracy']
        
        st.success(f"🏆 Best Model: **{best_model_name}** with {best_accuracy:.3f} accuracy")
        
        # Feature importance (for Random Forest)
        if 'Random Forest' in trained_models:
            st.subheader("Feature Importance (Random Forest)")
            rf_model = trained_models['Random Forest']['model']
            feature_cols = trained_models['Random Forest']['features']
            
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Feature Importance")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Individual Prediction":
        st.header("🔮 Individual Heart Disease Prediction")
        
        # Train models first
        with st.spinner("Loading prediction model..."):
            results, trained_models, _, _ = train_models(df)
        
        # Get the best model
        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
        best_model_name = results_df.iloc[0]['Model']
        best_model_info = trained_models[best_model_name]
        
        st.success(f"Using **{best_model_name}** model for prediction")
        
        # Input form
        st.subheader("Enter Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 20, 80, 50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
            chol = st.slider("Cholesterol", 100, 600, 250)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
            thalach = st.slider("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col3:
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
            slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
            ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
        
        # Make prediction
        if st.button("Predict Heart Disease", type="primary"):
            # Prepare input data
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                  thalach, exang, oldpeak, slope, ca, thal]])
            
            # Scale if necessary
            if best_model_info['scaler'] is not None:
                input_data = best_model_info['scaler'].transform(input_data)
            
            # Make prediction
            prediction = best_model_info['model'].predict(input_data)[0]
            probability = best_model_info['model'].predict_proba(input_data)[0]
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **High Risk** of Heart Disease")
                else:
                    st.success("✅ **Low Risk** of Heart Disease")
            
            with col2:
                disease_prob = probability[1] * 100
                st.metric("Disease Probability", f"{disease_prob:.1f}%")
            
            # Probability bar chart
            prob_df = pd.DataFrame({
                'Outcome': ['No Disease', 'Disease'],
                'Probability': probability
            })
            
            fig = px.bar(prob_df, x='Outcome', y='Probability',
                        title="Prediction Probabilities",
                        color='Outcome',
                        color_discrete_map={'No Disease': 'lightgreen', 'Disease': 'lightcoral'})
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

