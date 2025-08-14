# Sentiment-Analysis
# 📦 Amazon Reviews Sentiment Analysis

This project analyzes Amazon product reviews to classify them into **Negative**, **Neutral**, or **Positive** sentiments using **Natural Language Processing (NLP)** and **Machine Learning**.

## 📌 Overview
The workflow includes:
1. **Loading & Cleaning Data**  
   - Reads `amazon_reviews.csv`  
   - Removes missing reviews  
   - Labels sentiment based on star ratings

2. **Exploratory Data Analysis**  
   - Visualizes the distribution of sentiments using `seaborn`  
   - Cleans review text by removing special characters and converting to lowercase  

3. **Model Training & Evaluation**  
   - Splits the dataset into training and testing sets  
   - Uses **TF-IDF** vectorization to convert text into numerical features  
   - Trains a **Logistic Regression** classifier  
   - Evaluates model performance using a classification report and confusion matrix

4. **Visualization**  
   - Sentiment distribution bar plot  
   - Confusion matrix heatmap

## 🛠 Technologies Used
- **Python 3**
- [pandas](https://pandas.pydata.org/) for data manipulation
- [re](https://docs.python.org/3/library/re.html) for regex text cleaning
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) for visualization
- [scikit-learn](https://scikit-learn.org/stable/) for ML model training & evaluation

## 📂 Project Structure
