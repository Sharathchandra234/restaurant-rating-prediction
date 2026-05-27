# Restaurant Rating Prediction - End-to-End ML Project

## Project Overview
This project aims to predict the aggregate rating of restaurants based on various features such as location, cost, cuisine type, and service offerings. The goal is to build a production-ready machine learning pipeline that can accurately forecast restaurant ratings, providing valuable insights for both restaurant owners and customers.

## Business Problem
Restaurant ratings are crucial for customer decision-making and business success. By predicting ratings based on observable features, we can:
- Help restaurants understand which factors most influence their ratings
- Assist customers in making informed dining choices
- Provide data-driven insights for business improvement strategies

## Dataset
The dataset contains information about restaurants including:
- Restaurant ID, Name, Location (City, Address)
- Cuisine types
- Average cost for two people
- Service features (Online delivery, Table booking)
- Price range
- Aggregate rating (target variable)
- Votes
- Rating color and text

## Approach
1. **Exploratory Data Analysis (EDA)**: Comprehensive analysis of the dataset to understand distributions, relationships, and data quality issues.
2. **Feature Engineering**: Creation of meaningful features such as restaurant name frequency, cuisine popularity score, city-wise average ratings, cost efficiency metrics, and service quality indicators.
3. **Data Preprocessing**: Building a reusable pipeline for handling missing values, encoding categorical variables, scaling features, and feature selection.
4. **Model Training**: Training and comparing multiple advanced regression models including Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
5. **Hyperparameter Tuning**: Using GridSearchCV and RandomizedSearchCV to optimize the best-performing model.
6. **Model Evaluation**: Evaluating models using MAE, MSE, RMSE, R2 Score, cross-validation scores, residual analysis, and overfitting analysis.
7. **Explainability**: Using feature importance plots and SHAP analysis to understand which factors most influence restaurant ratings.
8. **Production-Level Improvements**: Implementing modular code structure, exception handling, logging, and reusable functions.
9. **Deployment**: Creating a Streamlit web app for interactive predictions and a REST API using FastAPI.

## Project Structure
```
restaurant_rating_prediction/
├── data/
│   └── Dataset.csv
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── modeling.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── train_model.py
│   ├── utils/
│   │   └── helpers.py
│   └── visualization/
│       └── plots.py
├── app/
│   ├── streamlit_app.py
│   └── api/
│       └── main.py
├── models/
│   ├── restaurant_rating_model.joblib
│   ├── preprocessor.joblib
│   └── feature_names.joblib
├── requirements.txt
└── README.md
```

## Features Engineered
- Restaurant name frequency
- Cuisine popularity score
- City-wise average ratings
- Cost efficiency metrics (rating per unit cost)
- Price category features
- Online delivery impact (binary)
- Table booking impact (binary)
- Weighted rating features
- Restaurant service quality indicators
- Log transformation of cost
- Cost categories (Low, Medium, High)
- Service score (combined online delivery and table booking)
- Is multivariate cuisine flag

## Models Compared
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-validation scores

## Results
[To be filled after model training]

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebooks in the `notebooks/` directory for exploratory analysis and model training
4. For deployment:
   - Streamlit app: `streamlit run app/streamlit_app.py`
   - FastAPI server: `uvicorn app.api.main:app --reload`

## Usage
### For Analysis and Modeling
1. Start with `notebooks/eda.ipynb` for exploratory data analysis
2. Proceed to `notebooks/feature_engineering.ipynb` for feature engineering steps
3. Finally, use `notebooks/modeling.ipynb` for model training, evaluation, and prediction

### For Deployment
- **Streamlit Web App**: Provides an interactive interface for users to input restaurant features and get predicted ratings.
- **REST API**: Exposes a `/predict` endpoint that accepts JSON input and returns predicted ratings.

## Future Improvements
1. Incorporate additional features such as review text analysis using NLP
2. Implement model monitoring and retraining pipeline
3. Add A/B testing framework for evaluating the impact of feature changes
4. Expand to include multi-class classification for rating categories
5. Integrate with real-time data sources for live predictions
6. Develop a recommendation system based on predicted ratings and user preferences

## Contact
[Your Name] - Machine Learning Intern at Cognifyz Technologies

## Acknowledgments
This project was developed as part of an internship at Cognifyz Technologies to demonstrate end-to-end machine learning engineering skills.