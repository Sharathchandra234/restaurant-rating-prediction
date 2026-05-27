# 🍽️ Restaurant Rating Prediction System

An end-to-end Machine Learning application that predicts restaurant ratings using restaurant features such as cuisine, pricing, delivery availability, and location.

Built using **Python, Scikit-Learn, FastAPI, and Streamlit**.

---

## 🚀 Features

- End-to-End ML Pipeline
- Interactive Streamlit Web App
- FastAPI REST API
- Feature Engineering
- Multiple ML Models Comparison
- Production-Style Project Structure
- Real-Time Restaurant Rating Prediction

---

## 🖥️ Demo

### Streamlit App
http://localhost:8501/

### FastAPI Swagger Docs

http://127.0.0.1:8000/docs

---

## 📊 Machine Learning Workflow

### Data Processing
- Missing Value Handling
- Feature Encoding
- Feature Engineering
- Scaling & Preprocessing

### Models Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

### Evaluation Metrics
- MAE
- MSE
- RMSE
- R² Score

---

## 🏗️ Project Structure

```bash
restaurant_rating_prediction/
│
├── app/
│   ├── api/
│   │   └── main.py
│   └── streamlit_app.py
│
├── data/
├── models/
├── notebooks/
├── src/
├── requirements.txt
└── README.md
⚡ Run Locally
Clone Repository
git clone https://github.com/Sharathchandra234/restaurant-rating-prediction.git
Install Dependencies
pip install -r requirements.txt
Run Streamlit App
streamlit run app/streamlit_app.py
Run FastAPI Server
uvicorn app.api.main:app --reload
🛠️ Tech Stack
Category	Technologies
Language	Python
ML	Scikit-Learn, XGBoost
Backend	FastAPI
Frontend	Streamlit
Data Processing	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Model Storage	Joblib
📌 Future Improvements
Docker Deployment
Cloud Hosting
NLP-based Review Analysis
Recommendation System
CI/CD Integration
👨‍💻 Author
Sharath Chandra

Machine Learning & AI Enthusiast

🔗 GitHub:
https://github.com/Sharathchandra234
