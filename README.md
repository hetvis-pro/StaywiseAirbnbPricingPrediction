# **Airbnb NYC Price Prediction — Machine Learning Project Report**

## **1. Introduction**

This project aims to develop machine learning models to predict the nightly price of Airbnb listings in New York City using the AB_NYC_2019 dataset.
The workflow includes:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Outlier Treatment
- Categorical Encoding
- Model Training
- Model Evaluation
- Hyperparameter Tuning
- MLflow Experiment Tracking

---

# # **2. Dataset Overview**

**Dataset:** `AB_NYC_2019.csv`
**Rows:** ~49,000
**Columns:** 16

### **Primary Columns Used**

| Type                | Columns                                                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Numerical**       | latitude, longitude, minimum_nights, number_of_reviews, reviews_per_month, calculated_host_listings_count, availability_365 |
| **Categorical**     | neighbourhood_group, neighbourhood, room_type                                                                               |
| **Target variable** | price                                                                                                                       |

---

# # **3. Data Cleaning**

### **3.1 Remove Unnecessary Columns**

Dropped columns that do not provide predictive value:

```
id, name, host_name, last_review
```

### **3.2 Missing Values**

- `reviews_per_month` → filled with **0**
- No other critical numerical missing values

### **3.3 Invalid Values**

- Removed rows where `price <= 0`
- Removed rows where `minimum_nights < 1`

---

# # **4. Outlier Detection & Treatment**

Price and minimum_nights had heavy-tailed distributions.

### **Method Used:**

**Upper-bound percentile trimming (95th percentile)**

```python
price_upper = df['price'].quantile(0.95)
mn_upper = df['minimum_nights'].quantile(0.95)

df = df[(df['price'] <= price_upper) &
        (df['minimum_nights'] <= mn_upper)]
```

**Reason:**
Preserves real low values but removes unrealistic high extremes (e.g., $5000/night).

---

# # **5. Exploratory Data Analysis (EDA)**

### **5.1 Price Distribution**

- Most prices fall between **$50–$160**
- Long tail on expensive listings

_(Insert histogram here)_

### **5.2 Price vs Room Type**

- Entire home/apt is the most expensive category
- Shared rooms are the cheapest

### **5.3 Minimum Nights**

- Majority require **1–5 nights**

### **5.4 Geographic Insight**

NYC areas with higher prices: Manhattan & Brooklyn.

### **5.5 Correlation Heatmap (Numeric Only)**

_(Insert heatmap image)_
Main insight:
Latitude/longitude correlate with price (location matters).

---

# # **6. Feature Engineering (Basic)**

Added meaningful features:

### **6.1 reviews_ratio**

```python
df['reviews_ratio'] = df['reviews_per_month'] / (df['number_of_reviews'] + 1)
```

### **6.2 is_superhost_like**

```python
df['is_superhost_like'] = (df['calculated_host_listings_count'] > 10).astype(int)
```

### **6.3 availability_category**

```python
# fully_booked / high_demand / medium_demand / low_demand
```

These improved data interpretability but were kept simple.

---

# # **7. Categorical Encoding**

Since models require numeric inputs:

- **OneHotEncoding used**
- Fit encoder **only on training data**
- Transform training and test sets separately
- Combined encoded categorical + numeric features manually

This avoids data leakage.

---

# # **8. Train/Test Split**

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

---

# # **9. Models Trained**

| Model             | MAE       | RMSE      |
| ----------------- | --------- | --------- |
| Linear Regression | ~36       | ~50       |
| Decision Tree     | ~46       | ~67       |
| **Random Forest** | **~33.6** | **~47.5** |
| XGBoost           | ~33.0     | ~46.9     |

### 🔥 **Best model: XGBoost / Random Forest (similar performance)**

(RF slightly more stable, XGBoost slightly better after tuning)

---

# # **10. Hyperparameter Tuning (XGBoost)**

Used `RandomizedSearchCV`

Best params found:

```
max_depth = 8
learning_rate = 0.05
n_estimators = 200
min_child_weight = 5
subsample = 0.7
colsample_bytree = 0.7
```

Result:

- Tuned MAE ~32.95
- Tuned RMSE ~46.80

Improvement was small → confirms model already well-fit.

---

# # **11. MLflow Tracking**

Logged for each model:

- MAE
- RMSE
- Hyperparameters
- Model artifacts
- Plots (true vs predicted, residuals, feature importance)

### **MLflow UI:**

Used to compare models and identify best-performing model.

![alt text](screenshots/mlflow/mlflowruns_evaluation.png)

### **Model Registry:**

Registered **XGBoost_best** as the production model.

![alt text](screenshots/mlflow/mlflow_model_registry.png)

---

# # **12. Final Deployment Preparation**

### Steps followed:

✔ Get best hyperparameters
✔ Retrain model on **full dataset** (train + test)
✔ Save final model (`.pkl`)
✔ Ready for deployment API / Streamlit

---

# # **13. Conclusion**

- **Goal:** Predict Airbnb listing prices
- **Best Model:** XGBoost (MAE ~32.9)
- **Key Contributors:**

  - Room type
  - Latitude/longitude
  - Neighborhood
  - Minimum nights
  - Host listing count

You created:

- Clean preprocessing workflow
- Basic but meaningful feature engineering
- Comparison of 4 ML models
- Hyperparameter-tuned model
- MLflow experiment tracking
- Model registry entry

---

# # **14. Future Improvements**

✔ Use advanced feature engineering (distances, review quality)
✔ Try neural networks
✔ Use SHAP explainability
✔ Deploy as API + Streamlit UI
✔ Use cross-validation instead of single split
