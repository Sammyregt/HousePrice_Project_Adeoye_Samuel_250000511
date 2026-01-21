# Project 3 – House Price Prediction System

## Project Overview

The goal of this project is to design and implement a **House Price Prediction System** using a machine learning algorithm.  

The system predicts house prices based on relevant features from the **“House Prices: Advanced Regression Techniques”** dataset.  

> **Note:** The dataset contains 79 features, but for this project, only the following nine features may be used:  
> `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`, `BedroomAbvGr`, `FullBath`, `YearBuilt`, `Neighborhood`, and `SalePrice`.  
> You may select **any six features** of your choice from these nine to build your model.

---

## PART A — Model Development

**Files:** `model_development.py` or `.ipynb`  

### Steps:

1. **Load the dataset.**

2. **Data Preprocessing:**
   - Handle missing values.
   - Feature selection.
   - Encode categorical variables (if applicable).
   - Feature scaling (where required).

3. **Model Implementation:**  
   Implement **any one** of the following algorithms:
   - Linear Regression
   - Random Forest Regressor
   - Support Vector Machine (SVR)
   - Prophet

4. **Model Training:**  
   Train the selected model using the preprocessed dataset.

5. **Model Evaluation:**  
   Evaluate the model using appropriate regression metrics, such as:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² (R squared)

6. **Model Saving:**  
   Save the trained model to disk using a method like **joblib** or **pickle**.  
   Ensure that the saved model can be **reloaded** without retraining.

---

## PART B — Web GUI Application

**Files:** `app.py` and `index.html`  

### Requirements:

Build a simple Web GUI that:

1. Loads the saved trained model.
2. Allows users to input house features.
3. Sends the input data to the model.
4. Displays the predicted house price.

### Permitted Technologies / Stack:

- Flask + HTML/CSS
- Streamlit
- FastAPI
- Django (not recommended)
- Gradio

---

## PART C — GitHub Submission

Upload the **entire project** to GitHub with the following structure:

/HousePrice_Project_yourName_matricNo/
│
├─ app.py
├─ requirements.txt
├─ /model/
│ ├─ model_building.ipynb
│ └─ house_price_model.pkl
├─ /static/
│ └─ style.css (optional if applicable)
└─ /templates/
└─ index.html (if applicable)

---

## PART D — Deployment Instructions

Deploy the Web GUI using **any one** of the following platforms:

- Render.com
- PythonAnywhere.com
- Streamlit Cloud
- Vercel
