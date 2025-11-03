#  Predictive Analysis of Heart Disease Using Machine Learning  

###  **Overview**
Heart disease remains one of the leading causes of death globally.  
This project demonstrates how **machine learning algorithms** can be used to predict the likelihood of heart disease based on key medical parameters.  

Using data preprocessing, exploratory data analysis (EDA), and multiple classification algorithms,  
this project identifies the most significant health indicators and builds a predictive model for early diagnosis.  

---

##  **Objective**
- Analyze patient health data to uncover hidden patterns.  
- Apply and compare multiple machine learning algorithms.  
- Determine the best-performing model for heart disease prediction.  
- Provide data-driven insights to assist in early diagnosis and prevention.  

---

##  **Tools and Technologies Used**

| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python |
| **Data Handling** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Machine Learning Models** | Scikit-learn |
| **Evaluation Metrics** | Accuracy, ROC-AUC, Confusion Matrix, Classification Report |
| **Development Environment** | Jupyter Notebook / IBM Skills Network |
| **Version Control** | GitHub |

---

##  **Dataset Information**

- **Source:** [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
- **Total Features:** 14 clinical attributes (age, sex, chest pain, cholesterol, blood pressure, etc.)  
- **Target Variable:**  
  - `1` → Presence of Heart Disease  
  - `0` → No Heart Disease  

---

##  **Exploratory Data Analysis (EDA)**

EDA was conducted to understand the structure and relationships in the data.  
It helps reveal correlations and significant patterns between health parameters and heart disease risk.

###  **Visual Insights Generated:**
1. **Age Distribution** — identifies the most affected age group.  
2. **Cholesterol Distribution** — shows cholesterol variation across patients.  
3. **Oldpeak Distribution** — visualizes ST depression levels.  
4. **Gender Distribution vs Heart Disease** — shows male/female comparison.  
5. **Chest Pain Type vs Heart Disease** — highlights chest pain patterns.  
6. **Correlation Heatmap** — reveals inter-feature relationships.  
7. **Cholesterol vs Age (colored by target)** — shows how risk grows with age.  
8. **Resting BP vs Max Heart Rate** — displays exercise effect on heart rate.  
9. **Pairplot for Key Features** — multi-variable relationship summary.

📊 These visuals helped identify **age, cholesterol, chest pain type, and max heart rate** as top predictors of heart disease.  

---

##  **Machine Learning Models Implemented**

Five ML algorithms were applied and compared:

| Model | Description |
|--------|--------------|
| **Logistic Regression** | Baseline model for binary classification. |
| **K-Nearest Neighbors (KNN)** | Distance-based non-parametric classifier. |
| **Decision Tree** | Tree-based model for easy interpretability. |
| **Random Forest** | Ensemble of decision trees for higher accuracy. |
| **Support Vector Machine (SVM)** | High-dimensional model for complex separation. |

All models were trained and tested using an **80:20 train-test split** after **feature scaling**.

---

##  **Model Evaluation & Comparison**

To evaluate the models’ performance, the following metrics were used:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score)**
- **ROC Curve and AUC Score**

###  **Model Performance Summary**

| Model | Accuracy | AUC | Remarks |
|--------|-----------|------|----------|
| Logistic Regression | 83% | 0.85 | Stable and interpretable |
| KNN | 80% | 0.82 | Sensitive to scaling |
| Decision Tree | 78% | 0.79 | Risk of overfitting |
| Random Forest | **88%** | **0.90** | Best overall performance |
| SVM | 86% | 0.88 | Excellent boundary separation |

** Best Model:** `Random Forest Classifier`  
** Key Insight:** Ensemble models (like Random Forest) outperform individual classifiers due to better generalization.

---

## **Evaluation Visuals**
- Model Accuracy Comparison Bar Chart  
- Confusion Matrices for Each Model  
- ROC Curve Comparison (All Models)  
- Final Accuracy Summary Graph  

These visual results validate that **Random Forest** consistently performs the best in predicting heart disease.

---

##  **Conclusion**

This project demonstrates how machine learning can be effectively applied to **healthcare prediction systems**.  
Through detailed analysis, visualization, and model evaluation, we achieved a reliable prediction model for heart disease.

###  **Key Findings**
- Data preprocessing and scaling are critical for better accuracy.  
- Visualization helps in understanding crucial medical risk factors.  
- Ensemble methods like Random Forest yield superior results.  

###  **Future Enhancements**
- Deploy the model using **Flask / Streamlit** for real-time patient prediction.  
- Integrate larger and more diverse datasets to improve accuracy.  
- Experiment with **Deep Learning models** (ANN, CNN) and **XGBoost** for enhanced performance.  

---

##  **Author**
**Name:** ARYA SINGH 

**Project Title:** Predictive Analysis of Heart Disease Using Machine Learning 

**Platform:** IBM Skills Network / GitHub  

**Year:** 2025  

---

##  **References**
- [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
- Python Official Documentation  
- Scikit-learn User Guide  
- IBM Data Science Professional Resources  

---

##  **Project Highlights**
✅ 7–8 EDA Graphs  
✅ 5 Machine Learning Models  
✅ ROC Curve & AUC Comparison  
✅ Confusion Matrices  
✅ Accuracy Summary & Conclusion  

---

> 🩺 *“Harnessing the power of data science to predict and prevent heart disease —  
because every heartbeat counts.”* 💖
