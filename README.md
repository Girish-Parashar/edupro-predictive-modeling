# 🎓 EduPro — Predictive Modeling for Course Demand & Revenue Forecasting

## 📌 Project Overview
A complete Data Science project analyzing EduPro's online learning platform data 
to predict course enrollment demand and revenue using Machine Learning.

## 🎯 Objectives
- Predict course enrollment count
- Predict course revenue
- Predict category-level revenue
- Build an interactive analytics dashboard

## 📊 Dataset
| Sheet | Records |
|---|---|
| Courses | 60 |
| Teachers | 60 |
| Transactions | 10,000 |
| Users | 3,000 |

## 🤖 Best Models
| Target | Model | R² Score |
|---|---|---|
| Enrollment Count | Random Forest | -0.144 (data limited) |
| Course Revenue | Ridge Regression | 0.983 ✅ |
| Category Revenue | Gradient Boosting | 0.893 ✅ |

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, Joblib
- Plotly, Streamlit
- Jupyter Notebook, VS Code

## 🚀 Run the Dashboard
```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

## 📁 Project Structure
```
├── data/          → CSV datasets
├── notebooks/     → Jupyter EDA + ML notebook
├── models/        → Trained .pkl model files
├── app/           → Streamlit dashboard
└── report/        → Research paper (.docx)
```

## 🏆 Project By
Unified Mentor — Data Science Internship Project 2025
```

---

## 📦 One More — `requirements.txt`

Create `requirements.txt` in your `project\` root folder:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
streamlit
joblib
openpyxl
```

---

## ✅ Final Checklist Before Pushing
```
project/
├── .gitignore        ← CREATE THIS
├── README.md         ← CREATE THIS  
├── requirements.txt  ← CREATE THIS
├── data/             ← PUSH ✅
├── notebooks/        ← PUSH ✅
├── models/           ← PUSH ✅
├── app/              ← PUSH ✅
└── report/           ← PUSH ✅