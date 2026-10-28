# INX Future Inc. Employee Performance Analysis

## Project Overview
This project analyzes employee data to understand performance ratings and identify the factors associated with employee performance at INX Future Inc.

A Machine Learning classification model is used to predict employee performance ratings based on employee-related features.

## Objectives
- Analyze employee performance data.
- Explore factors associated with employee performance.
- Build a Machine Learning classification model.
- Predict employee performance ratings through a Streamlit web application.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Plotly

## Machine Learning Model
**Selected Model:** Decision Tree Classifier

The model was tuned using:
- `max_depth = 5`
- `min_samples_leaf = 2`
- `min_samples_split = 10`

### Model Evaluation
| Metric | Score |
|---|---:|
| Accuracy | 93.75% |
| Macro F1-score | 0.8996 |
| Macro Recall | 0.8661 |

### Important Features
- Employee Environment Satisfaction
- Employee Last Salary Hike Percentage
- Years Since Last Promotion

## Project Files
- `Employee_Performance_Analysis.ipynb` — Data analysis and model development.
- `app.py` — Streamlit application.
- `final_employee_performance_model.pkl` — Saved trained model.
- `requirements.txt` — Required Python libraries.

## Run the Application Locally

1. Clone or download this repository.
2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:

   ```bash
   python -m streamlit run app.py
   ```

## Disclaimer
The model provides an analytical estimate of employee performance ratings. Predictions should not be used as the sole basis for employee evaluation or employment decisions.

## Author
Saniya Santosh Lokhande

Data Science | Data Analytics | Machine Learning
