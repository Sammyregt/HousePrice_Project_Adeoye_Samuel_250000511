# Project 3 – House Price Prediction System

## Project Overview

You are required to design and implement a **House Price Prediction System** using a machine learning algorithm.

The system predicts house prices based on relevant features from the **“House Prices: Advanced Regression Techniques”** dataset.

The dataset has 79 features, but only the following nine may be used for this project:

- OverallQual  
- GrLivArea  
- TotalBsmtSF  
- GarageCars  
- BedroomAbvGr  
- FullBath  
- YearBuilt  
- Neighborhood  
- SalePrice  

> 👉 To build the model, select any six (6) features of your choice from the recommended nine (excluding **SalePrice**, which is the target variable).

---

## PART A — Model Development

**File:** `model_development.py` or `model_building.ipynb`

You are required to:

- Load the dataset  
- Perform data preprocessing, including:  
  - Handling missing values  
  - Feature selection  
  - Encoding categorical variables (if applicable)  
  - Feature scaling (where required)  

- Implement any one of the following algorithms:  
  - Linear Regression  
  - Random Forest Regressor  
  - Support Vector Regression (SVR)  
  - Prophet  

- Train the model using the dataset  
- Evaluate the model using appropriate regression metrics, such as:  
  - MAE  
  - MSE  
  - RMSE  
  - R²  

- Save the trained model to disk using an appropriate method (e.g., Joblib or Pickle)  
- Ensure the saved model can be reloaded without retraining  

---

## PART B — Web GUI Application

**Files:** `app.py` and `index.html` (if applicable)

Build a simple Web-based Graphical User Interface (GUI) that:

- Loads the saved trained model  
- Allows users to input house features  
- Sends the input data to the model  
- Displays the predicted house price  

---

## Permitted Technologies / Stack

- Flask + HTML/CSS  
- Streamlit  
- FastAPI  
- Django (not recommended)  
- Gradio  

---

## PART C — GitHub Submission

Upload the entire project to GitHub using the structure below:

/HousePrice_Project_yourName_matricNo/
|
|- app.py
|- requirements.txt
|
|- /model/
| |- model_building.ipynb
| |- house_price_model.pkl
|
|- /static/
| |- style.css (optional, if applicable)
|
|- /templates/
|- index.html (if applicable)


---

## PART D — Deployment Instructions

Deploy the Web GUI using any one of the following platforms:

- Render.com  
- PythonAnywhere.com  
- Streamlit Cloud  
- Vercel  
