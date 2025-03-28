

# 🏠 Real Estate Sales Analysis (2017–2022)

## Project Name: **"SmartValue: Real Estate Sales Insights (2017–2022)"**

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Objective](#project-objective)
3. [Dataset Description](#dataset-description)
4. [Technologies Used](#technologies-used)
5. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - Town-level Insights
   - Property Type Performance
   - Sales Trends Over Time
7. [Predictive Modeling](#predictive-modeling)
   - Modeling Question
   - Model Pipeline
   - Evaluation Metrics
   - Predictions vs. Actuals
   - Feature Importance
8. [Interpretation & Key Takeaways](#-interpretation--key-takeaways)
9. [Conclusion & Recommendations](#-conclusion--recommendations)
10. [Final Thoughts](#-final-thoughts)
11. [References](#-references)

---

### 📖 Project Overview

This project involves the analysis of cleaned real estate sales data from the years **2017 to 2022**, focusing on various aspects such as property types, sale prices, assessed values, and sales ratios across different towns. The dataset provides useful information for understanding real estate trends in Connecticut, USA.

The project aims to derive actionable insights that can support stakeholders—including homebuyers, investors, analysts, and government entities in making data-driven decisions in the housing market.

---

### Project Objectives

- Understand real estate market trends between 2017 and 2022.
- Analyze variations in **Sale Amount**, **Sales Ratio**, and **Assessed Value** across towns.
- Identify top towns based on total sales and average property values.
- Discover patterns between **Property Type**, **Residential Type**, and sales performance.
- Evaluate anomalies or areas with high variance in sales and assessment.

---

### Technologies Used

- **Python**
- **Pandas** (Data Cleaning, Analysis)
- **NumPy** (Numerical Operations)
- **Matplotlib & Seaborn** (Visualization - optional extension)
- **Jupyter Notebook**
- **CSV file format**

---

### 📂 Dataset Source

- **URL:** [Data.gov Real Estate Sales](https://catalog.data.gov/dataset/real-estate-sales-2001-2018)
- **File Used:** `Cleaned_Real_Estate_Sales_2017-2022_GL.csv`


---

### Expected Insights

1. Towns with the **highest and lowest sale amounts**

![top_10_towns_by_sales](https://github.com/user-attachments/assets/1305200c-b0b1-4369-8beb-0e4f55da5246)

#### Top 10 Towns by Total Sale Amount
- **Danbury** overwhelmingly leads with over **$22 million** in sales.
- Followed by **Norwalk**, **Berlin**, and **Bridgeport**, though their totals are significantly lower.
- A steep drop is seen after the top few towns, indicating **concentrated high-value activity**.

![bottom_10_towns_by_sales](https://github.com/user-attachments/assets/8f508de4-8221-4cd7-81bc-853500a8606f)

#### Bottom 10 Towns by Total Sale Amount
- **Bethlehem**, **Thomaston**, and **Prospect** recorded the **lowest total sales**, under **$90,000**.
- Even the highest in this group, like **New London** and **Thompson**, remain below **$150,000**.
- Indicates **low transaction volume** or smaller market size in these towns.

2. **Average Sales Ratio** by Property Type

![avg_sales_ratio_by_property_type](https://github.com/user-attachments/assets/c5b41ddf-eb4c-4a09-a973-ee27cc6bd79b)

### Summary: Average Sales Ratio by Property Type (2017–2022)

- **Four Family** and **Three Family** properties had the **highest average sales ratios** (above 1.0), indicating they often sold above assessed value.
- **Two Family** and **Single Family** followed with strong ratios.
- **Condo** and **Residential** types had the **lowest sales ratios**, suggesting they sold below their assessed values on average.

#### 💡 Insight:
Multi-family properties (3–4 units) generally offer **better return relative to assessed value** than single-family or condo units.


3. Correlation between **Assessed Value** and **Sale Amount**
![assessed_vs_sale_amount](https://github.com/user-attachments/assets/89cd0634-2393-4091-a80f-c13ed1018590)

### Summary: Assessed Value vs. Sale Amount

- The scatter plot shows a **positive correlation** between **Assessed Value** and **Sale Amount**.
- Higher assessed values generally lead to higher sale prices, although **some variability and outliers** exist.

#### Insight:
Assessed value is a **strong predictor** of sale amount, making it useful for pricing models and valuation strategies.


4. Top-performing **Residential Types**
![avg_sale_by_residential_type](https://github.com/user-attachments/assets/59466747-5157-41f6-a46d-384c8f9cc611)

### Summary: Average Sale Amount by Residential Type (2017–2022)

- **Single Family** homes had the **highest average sale amount**, followed by **Condos**.
- **Two Family** and **Three Family** homes showed moderate averages.
- **Four Family** properties had the **lowest average sale amount**.

#### Insight:
Single-family and condo properties tend to command **higher market value**, while larger multi-family units are priced lower on average.

5. Frequency and patterns of sales by **year** and **town**
![top5_town_sales_trend](https://github.com/user-attachments/assets/10931625-2df3-4072-8303-c0639c59e3d9)

### Summary of Yearly Sales Trend for Top 5 Towns (2017–2022)

- **Danbury** shows a sharp rise in sales from 2018 to 2020, maintaining high activity through 2022.
- **Berlin** and **Norwalk** saw increasing trends, with Berlin peaking in 2022.
- **Bridgeport** and **New Haven** remained relatively stable with lower activity.

####  Insight:
Danbury consistently leads in yearly real estate transactions, indicating a strong and active housing market.


6. Outliers or irregular sales based on **Sales Ratio**
![sales_ratio_outliers](https://github.com/user-attachments/assets/c60c73ff-7add-4465-b91e-98ab9219f520)

###  Summary: Sales Ratio Distribution (Boxplot)

- Most properties had a **Sales Ratio between 0.25 and 1.75**.
- The **median** was around **0.6**, indicating many properties sold **below assessed value**.
- A few **high outliers** exist, where properties sold well **above their assessed value**.

####  Insight:
The presence of high sales ratio outliers may indicate **premium properties**, bidding wars, or under-assessed valuations.

---

### Data Cleaning & Preprocessing Summary

- **Missing values:** Removed
- **Duplicates:** Dropped
- **Data types:** Converted (`Date Recorded` to datetime, others cleaned)
- **Outliers:** Removed using IQR method
- **Saved as:** `Cleaned_Real_Estate_Sales_2017-2022_GL.csv`

---

### Data Analysis (Initial Observations)

- `Sale Amount` and `Assessed Value` are numeric and ideal for correlation analysis.
- `Sales Ratio` (Sale Amount / Assessed Value) can be analyzed across towns and property types.
- `Town`, `Property Type`, and `Residential Type` are useful for categorical segmentation.
- `Date Recorded` can be parsed for time-series analysis or yearly trends.

---

###  Predictive Data Modeling

**How accurately can we predict the sale amount of real estate properties in Connecticut using historical data from 2017 to 2022, based on features such as assessed value, property type, residential type, town, and sale year?**

![predicted_vs_actual](https://github.com/user-attachments/assets/d24985b0-7902-4185-b629-19415000e374)

### Predicted vs. Actual Sale Amount

- Most predicted values are fairly close to the red diagonal line, indicating decent model performance.
- Some scatter around the line suggests **prediction variance**, especially for higher-priced properties.

#### Insight:
The model performs reasonably well for mid-range sales, but **underestimates or overestimates outliers**, which is expected due to limited features and sample size.


![feature_importance](https://github.com/user-attachments/assets/be47330b-fb79-4c8c-b474-ae1eb78b1576)

### Top 10 Important Features for Sale Amount Prediction

- **Assessed Value** is by far the most influential predictor.
- **Property Type (Residential)** and **Year** also have meaningful impact.
- Specific towns like **Danbury**, **North Branford**, and **Norwalk** contribute moderately.

#### Insight:
The model heavily relies on **Assessed Value**, reaffirming its importance in estimating sale price. Town-specific and property-type variations provide additional predictive value.

### Data Analysis
**Load and Prepare the Data**
```python
# Load the cleaned dataset
df = pd.read_csv('Cleaned_Real_Estate_Sales_2017-2022_GL.csv')

# Convert 'Date Recorded' to datetime and extract year
df['Date Recorded'] = pd.to_datetime(df['Date Recorded'], errors='coerce')
df['Year'] = df['Date Recorded'].dt.year
```

**Define Features and Target**
```python
# Define feature columns and target column
features = ['Assessed Value', 'Property Type', 'Residential Type', 'Town', 'Year']
target = 'Sale Amount'

X = df[features]
y = df[target]
```
**Create Preprocessing Pipeline**

```python

# Identify categorical features for encoding
categorical_features = ['Property Type', 'Residential Type', 'Town']

# Column transformer to one-hot encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Keep other columns (numerical)
)
```

**Build the Random Forest Model Pipeline**

```python
# Create pipeline with preprocessing + model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
```

**Split the Data for Training and Testing**
```python
# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Train the Model**
```python
# Fit the model
model.fit(X_train, y_train)
```
### 🧱 Model Pipeline Structure

```python
Pipeline(steps=[
    ('preprocessor',
     ColumnTransformer(
         remainder='passthrough',
         transformers=[
             ('cat',
              OneHotEncoder(handle_unknown='ignore'),
              ['Property Type', 'Residential Type', 'Town']
             )
         ]
     )
    ),
    ('regressor', RandomForestRegressor(random_state=42))
])
```

**Make Predictions and Evaluate**

```python
# Predict on test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Display metrics
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R² Score: {r2:.2f}")
```
**Model Evaluation Metrics (Random Forest Regressor)**

| Metric                         | Value        |
|-------------------------------|--------------|
| Mean Absolute Error (MAE)     | $59,620.40   |
| Root Mean Squared Error (RMSE)| $93,050.27   |
| R² Score                      | 0.27         |


### Interpretation & Key Takeaways

-  **Assessed Value is King**: The most important driver of sale price — consistently dominating the prediction model.
-  **Multi-family homes outperform on ROI**: Property types like Four-Family and Three-Family homes had the highest average sales ratios.
-  **Danbury leads the market**: With the highest total sales volume and transaction frequency across all years.
-  **Prediction model performs reasonably**: With an R² of 0.27, the model is functional but leaves room for improvement with additional features.

---

### Conclusion & Recommendation

- The analysis reveals clear patterns across **towns, property types, and sale behaviors** from 2017–2022.
- Even with basic features, we were able to train a working model to **predict sale amounts**, which can assist stakeholders in price estimation.
- Data-driven tools have value — especially for small municipalities, developers, and investors who want a starting point for property assessment.

---

### Recommendations

1. **Enhance the dataset**: Include additional features like square footage, number of rooms, building age, and neighborhood ratings to improve predictions.
2. **Focus on rising towns**: Towns like Danbury, Berlin, and Norwalk show strong activity — ideal for deeper investment or trend forecasting.
3. **Develop a pricing tool**: Convert the model into a web app to help local buyers/sellers evaluate fair property prices.
4. **Incorporate time series modeling**: Predict future market behavior by including time-based trends and seasonal sales patterns.

---

### Final Thoughts

This project showcases the potential of **public real estate data** when properly cleaned, visualized, and modeled.  
It highlights both the **power and limitations** of basic predictive modeling and lays a foundation for building **more advanced tools** to support real estate strategy.

> “Without data, you're just another person with an opinion.” – W. Edwards Deming

---


### 🔗 References

1. **Dataset Source**  
   [Real Estate Sales 2001–2018 – Data.gov](https://catalog.data.gov/dataset/real-estate-sales-2001-2018)  
   - Used for analyzing property sales across Connecticut from 2017 to 2022.

2. **Python Libraries & Tools**  
   - `pandas` – Data manipulation and analysis  
   - `numpy` – Numerical computations  
   - `matplotlib` & `seaborn` – Data visualization  
   - `scikit-learn` – Machine learning and model evaluation

3. **Inspirations & Best Practices**  
   - scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)  
   - Data storytelling and visualization standards from [datavizproject.com](https://datavizproject.com)

4. **Quotes & Ideas**  
   - W. Edwards Deming: *“Without data, you're just another person with an opinion.”*









