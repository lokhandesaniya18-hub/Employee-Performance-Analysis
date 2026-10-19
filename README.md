# INX Future Inc. Employee Performance Analysis

## Project Overview

This project analyzes employee performance at INX Future Inc. using Python and Machine Learning. The main goal is to understand employee performance, compare performance across departments, identify important factors, and predict employee performance ratings.

## Project Objectives

- Analyze employee performance department-wise.
- Understand factors related to employee performance.
- Identify the top 3 important features.
- Build and evaluate Machine Learning models.
- Suggest ways to improve employee performance.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow

1. Data Loading and Inspection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Data Visualization
5. Model Building
6. Model Evaluation
7. Hyperparameter Tuning
8. Feature Importance Analysis
9. Business Recommendations

## Key Findings

- Performance Rating 3 was the most common rating.
- The Development department had the highest average performance rating.
- The Finance department had the lowest average performance rating.
- Employee satisfaction, salary hikes, and promotion opportunities were important factors related to performance.

## Top 3 Important Features

1. Employee Environment Satisfaction
2. Employee Last Salary Hike Percentage
3. Years Since Last Promotion

## Machine Learning Model

Multiple classification models were compared, including Logistic Regression, Decision Tree, Random Forest, SVM, KNN, and Gradient Boosting.

The **Tuned Decision Tree Classifier** was selected as the final model.

### Model Performance

- **Accuracy:** 93.75%
- **Macro F1-score:** 0.8996
- **Macro Recall:** 0.8661

The model performed best for Performance Rating 3, with 98% recall. It achieved 77% recall for Rating 4.

## Business Recommendations

1. Improve employee satisfaction by collecting regular feedback.
2. Keep salary hikes fair and transparent.
3. Provide clear promotion and career development opportunities.
4. Encourage a healthy work-life balance.
5. Review model predictions to improve the identification of Rating 4 employees.

## Conclusion

This project helped analyze employee performance and identify important factors related to performance ratings. The Tuned Decision Tree Classifier achieved 93.75% accuracy. The findings can help HR teams understand employee performance patterns and make better data-driven decisions.
