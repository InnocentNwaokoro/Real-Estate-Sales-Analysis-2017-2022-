

# 🏠 Real Estate Sales Analysis (2017–2022)

## 📌 Project Name: **"SmartValue: Real Estate Sales Insights (2017–2022)"**

---

## 📖 Introduction

This project involves the analysis of cleaned real estate sales data from the years **2017 to 2022**, focusing on various aspects such as property types, sale prices, assessed values, and sales ratios across different towns. The dataset provides useful information for understanding real estate trends in Connecticut, USA.

The project aims to derive actionable insights that can support stakeholders—including homebuyers, investors, analysts, and government entities—in making data-driven decisions in the housing market.

---

## 🎯 Objectives

- Understand real estate market trends between 2017 and 2022.
- Analyze variations in **Sale Amount**, **Sales Ratio**, and **Assessed Value** across towns.
- Identify top towns based on total sales and average property values.
- Discover patterns between **Property Type**, **Residential Type**, and sales performance.
- Evaluate anomalies or areas with high variance in sales and assessment.

---

## 🛠 Technologies Used

- **Python**
- **Pandas** (Data Cleaning, Analysis)
- **NumPy** (Numerical Operations)
- **Matplotlib & Seaborn** (Visualization - optional extension)
- **Jupyter Notebook**
- **CSV file format**

---

## 📂 Dataset Source

- **URL:** [Data.gov Real Estate Sales](https://catalog.data.gov/dataset/real-estate-sales-2001-2018)
- **File Used:** `Cleaned_Real_Estate_Sales_2017-2022_GL.csv`
- **Records:** 223 cleaned records
- **Features (Columns):** 14

---

## 🔍 Expected Insights

- Towns with the **highest and lowest sale amounts**
- **Average Sales Ratio** by Property Type
- Correlation between **Assessed Value** and **Sale Amount**
- Top-performing **Residential Types**
- Frequency and patterns of sales by **year** and **town**
- Outliers or irregular sales based on **Sales Ratio**

---

## 🧼 Data Cleaning & Preprocessing Summary

- **Missing values:** Removed
- **Duplicates:** Dropped
- **Data types:** Converted (`Date Recorded` to datetime, others cleaned)
- **Outliers:** Removed using IQR method
- **Saved as:** `Cleaned_Real_Estate_Sales_2017-2022_GL.csv`

---

## 📊 Data Analysis (Initial Observations)

- `Sale Amount` and `Assessed Value` are numeric and ideal for correlation analysis.
- `Sales Ratio` (Sale Amount / Assessed Value) can be analyzed across towns and property types.
- `Town`, `Property Type`, and `Residential Type` are useful for categorical segmentation.
- `Date Recorded` can be parsed for time-series analysis or yearly trends.


