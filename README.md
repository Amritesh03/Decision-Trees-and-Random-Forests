# Task 5 — Decision Trees & Random Forests

## 📌 Objective
Implement and compare **Decision Tree Classifier** and **Random Forest Classifier** on the Heart Disease dataset.  
Learn about tree-based models, overfitting control, and feature importance interpretation.

## 📂 Dataset
- **Name:** Heart Disease Dataset  
- **Source:** [Kaggle Link](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
- **Target Variable:** `target` (1 = disease, 0 = no disease)

## 🛠 Tools & Libraries
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib
- graphviz / pydotplus (for tree visualization)
- joblib (for saving models)

## 📊 Steps Performed
1. **Data Loading & Exploration**
   - Checked shape, column names, and target distribution.
2. **Preprocessing**
   - Imputed missing values.
   - Scaled numeric features & one-hot encoded categorical features.
3. **Decision Tree Classifier**
   - Trained baseline model.
   - Visualized decision tree.
   - Controlled overfitting using `max_depth`.
4. **Random Forest Classifier**
   - Trained baseline model.
   - Compared performance with Decision Tree.
5. **Feature Importance**
   - Extracted & visualized top features from Random Forest.
6. **Evaluation**
   - Accuracy, Classification Report, Confusion Matrix, Cross-validation.
7. **Hyperparameter Tuning**
   - Used `GridSearchCV` for both models.
8. **Model Saving**
   - Saved pipelines and best model to `models/` folder.

## 📈 Results
- **Decision Tree Accuracy:** ~XX% (baseline)
- **Random Forest Accuracy:** ~YY% (baseline)
- Random Forest performed better and was less prone to overfitting.
- Top important features: (Example) `cp`, `thalach`, `oldpeak`, ...

## 📷 Visualizations
- Decision Tree plot (matplotlib/Graphviz)
- Feature Importance bar chart
- Train vs Test Accuracy for varying `max_depth`

# Run script
python task5_decisiontree_randomforest.py
