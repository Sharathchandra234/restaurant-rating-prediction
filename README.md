# 🍽️ Restaurant Rating Prediction System

> End-to-End Machine Learning Engineering Project using FastAPI, Streamlit, and Scikit-Learn

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Machine Learning](https://img.shields.io/badge/ML-ScikitLearn-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🚀 Live Demo

### Streamlit Web App
🔗 Add your deployed Streamlit link here

Example:
```bash
https://restaurant-rating-predictor.streamlit.app


FastAPI Swagger Docs
http://127.0.0.1:8000/docs

📌 Project Overview

This project predicts restaurant ratings using Machine Learning based on restaurant features such as:

Cuisine types
Cost for two
Location
Online delivery availability
Table booking
Price range
Customer engagement metrics

The system was designed as a production-style ML application with:

modular architecture
reusable preprocessing pipeline
REST API serving
interactive Streamlit frontend
deployment-ready structure
✨ Key Features

✅ End-to-End ML Pipeline
✅ Feature Engineering
✅ FastAPI REST Backend
✅ Interactive Streamlit UI
✅ Modular Project Structure
✅ Multiple ML Models Comparison
✅ Hyperparameter Tuning
✅ Production-Oriented Design
✅ Real-Time Rating Prediction

🧠 Machine Learning Workflow
1️⃣ Exploratory Data Analysis
Missing value analysis
Correlation analysis
Distribution analysis
Feature relationships
Outlier detection
2️⃣ Feature Engineering

Custom engineered features include:

Cuisine popularity score
Cost efficiency metrics
City-wise average ratings
Service quality indicators
Delivery & booking impact
Multi-cuisine flags
3️⃣ Models Trained
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
4️⃣ Evaluation Metrics
MAE
MSE
RMSE
R² Score
Cross Validation
🏗️ Project Architecture
restaurant_rating_prediction/
│
├── app/
│   ├── api/
│   │   └── main.py
│   └── streamlit_app.py
│
├── data/
│   └── Dataset.csv
│
├── models/
│   ├── restaurant_rating_model.joblib
│   ├── preprocessor.joblib
│   └── feature_engineer.joblib
│
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── preprocessing/
│   └── visualization/
│
├── requirements.txt
└── README.md
🖥️ Streamlit Application

The Streamlit frontend allows users to:

Enter restaurant details
Predict restaurant ratings
Interact with the ML system visually
Simulate real-world restaurant evaluation
⚡ FastAPI Backend

REST API endpoints built using FastAPI.

Run API
uvicorn app.api.main:app --reload
Swagger Documentation
http://127.0.0.1:8000/docs
📦 Installation
Clone Repository
git clone https://github.com/Sharathchandra234/restaurant-rating-prediction.git
Move Into Project
cd restaurant-rating-prediction
Install Dependencies
pip install -r requirements.txt
▶️ Run Streamlit App
streamlit run app/streamlit_app.py
📊 Tech Stack
Category	Technologies
Language	Python
ML	Scikit-Learn, XGBoost
Data	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Backend	FastAPI
Frontend	Streamlit
Model Storage	Joblib
📈 Future Improvements
Docker Deployment
CI/CD Pipeline
Cloud Deployment
Real-Time Prediction API
NLP-Based Review Analysis
Recommendation System
👨‍💻 Author
Sharath Chandra

Machine Learning & AI Engineering Enthusiast

🔗 GitHub:
https://github.com/Sharathchandra234
