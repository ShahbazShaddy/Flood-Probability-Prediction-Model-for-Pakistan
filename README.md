# Flood Probability Prediction Model for Pakistan

## Project Overview
This project develops a data-driven machine learning model to predict flood probability in Pakistan, addressing the critical need for proactive disaster mitigation. Utilizing historical environmental data, the model analyzes factors such as monsoon intensity, topography drainage, deforestation, urbanization, and river management to forecast flood risk with high accuracy. The project includes exploratory data analysis (EDA), data preprocessing, model training, evaluation, and serialization for potential deployment.

---

## Objectives
- Build a predictive model to identify flood risk using environmental and infrastructural data.  
- Compare multiple regression algorithms to determine the most effective model for flood prediction.  
- Enable proactive decision-making for disaster preparedness and mitigation in flood-prone regions.  

---

## Dataset
The dataset consists of over 1.1 million records with 21 features, including:  
- `MonsoonIntensity`, `TopographyDrainage`, `RiverManagement`, `Deforestation`, `Urbanization`, `ClimateChange`, etc.  
- **Target variable:** `FloodProbability` (continuous value between 0 and 1).  
- **Source:** Kaggle dataset (`train.csv`, `test.csv`).  

---

## Methodology

### Data Preprocessing
- Loaded and inspected the dataset using Pandas.  
- Dropped irrelevant columns (e.g., `id`) and split data into features (`X`) and target (`y`).  
- Applied train-test splitting (80-20 split) and standardized features using `StandardScaler`.  

### Exploratory Data Analysis (EDA)
- Analyzed dataset structure, summary statistics, and distributions using Pandas, Matplotlib, and Seaborn.  
- Identified feature correlations and target variable distribution.  

### Modeling
Implemented and trained multiple regression models:  
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- K-Nearest Neighbors (KNN) Regressor  

Evaluated models using:  
- **Mean Absolute Error (MAE)**  
- **Root Mean Squared Error (RMSE)**  
- **R² score**  

### Model Serialization
- Saved trained models using `Joblib` for future use or deployment.  

---

## Results
The models were evaluated on a validation set, with the following performance metrics:

| Model              | MAE     | RMSE    | R²     |
|--------------------|---------|---------|--------|
| Linear Regression  | 0.0158  | 0.0201  | 0.8449 |
| Ridge Regression   | 0.0158  | 0.0201  | 0.8449 |
| Lasso Regression   | 0.0161  | 0.0202  | 0.8427 |
| Gradient Boosting  | 0.0205  | 0.0249  | 0.7623 |
| KNN Regressor      | 0.0237  | 0.0293  | 0.6700 |
| Random Forest      | 0.0244  | 0.0298  | 0.6584 |

**Key Insight:** Linear Regression and Ridge Regression outperformed other models, achieving the highest R² score of **0.845**, indicating strong predictive capability.  

---

## Technologies and Tools
- **Programming Language:** Python  
- **Libraries:**  
  - Data Manipulation: Pandas, NumPy  
  - Visualization: Matplotlib, Seaborn  
  - Machine Learning: Scikit-learn (regression models, preprocessing, metrics)  
  - Model Serialization: Joblib  
- **Environment:** Jupyter Notebook  
- **Hardware:** NVIDIA Tesla T4 GPU (via Kaggle environment)  

---

## Installation and Setup
```bash
# Clone the repository
git clone <repository_url>

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib
