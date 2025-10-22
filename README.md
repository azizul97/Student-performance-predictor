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
## Screenshots
<img width="1919" height="1017" alt="Screenshot 2025-10-22 131402" src="https://github.com/user-attachments/assets/64b5ca01-3629-48d3-a8e4-d55e1e626a1d" />
<img width="1919" height="1020" alt="Screenshot 2025-10-22 131430" src="https://github.com/user-attachments/assets/b29c052d-b6c0-4cdf-98b8-fe05a46e62d3" />
<img width="1919" height="1022" alt="Screenshot 2025-10-22 131527" src="https://github.com/user-attachments/assets/7568b589-c89f-4ccd-a0e2-46cf20ac8e8c" />
<img width="1919" height="1023" alt="Screenshot 2025-10-22 131546" src="https://github.com/user-attachments/assets/371daa90-acb8-4cb5-8d9f-60acc0eb9dee" />
<img width="1919" height="1019" alt="Screenshot 2025-10-22 131644" src="https://github.com/user-attachments/assets/74055d46-dd89-4686-a705-09e0b3258982" />


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

