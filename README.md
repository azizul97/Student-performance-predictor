🎓 Predicting Student Performance using Machine Learning

This project applies **data analytics and machine learning** to predict students' academic performance based on various socio-economic and academic features.  
The goal is to identify key factors influencing success and enable early intervention for struggling students.

---

## 📊 Project Overview

- **Objective:** Predict student performance (pass/fail or score) using demographic and behavioral data.  
- **Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Model Used:** Random Forest Classifier (best performing model after comparison)  
- **Output:** Visualized feature importance, ROC curve, and trained model file.

---

## 🧠 Key Features

✅ Data cleaning and preprocessing  
✅ Exploratory Data Analysis (EDA)  
✅ Model training & evaluation (accuracy, ROC AUC, confusion matrix)  
✅ Feature importance ranking  
✅ Exported trained model (`.joblib`) for deployment  

---

## 📁 Project Structure

student-performance-prediction/
│
├── datastudents_performance.csv.csv # Dataset used
├── predict_student_performance.py # Main project script
├── outputs/ # Model and visual outputs
│ ├── model.joblib
│ ├── feature_importances.csv
│ └── roc_random_forest.png
├── requirements.txt # Dependencies
└── README.md # Documentation

## Install dependencies

pip install -r requirements.txt


## Run the script

python predict_student_performance.py


## Check outputs

Model & visuals will be saved inside the /outputs/ folder.

## 📈 Results

Best Model: Random Forest Classifier

Accuracy: ~85% (depending on dataset version)

Key Features: Study time, parental education, and past failures.

## 👨‍💻 Author
Baig Azizul Hakim
📍 Aspiring AI & Frontend Developer | Data Enthusiast
