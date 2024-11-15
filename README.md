# Coffee Reviews Analysis and Model Training Dashboard using Streamlit  

This Streamlit dashboard is designed to analyze and model a comprehensive coffee dataset that includes details on coffee name, type, origin, and descriptive reviews. The project focuses on **data preprocessing** and **exploratory data analysis (EDA)** to reveal key insights, which serve as the foundation for various machine learning models to **predict prices, analyze sentiment in reviews, provide coffee recommendations**, and **group similar coffees**. This project focuses on predicting coffee prices, analyzing sentiment in reviews, providing coffee recommendations, and clustering coffees based on key characteristics.

## 👥 Group Members
- **Bunag, Annika**
- **Chua, Denrick**
- **Mallillin, Gena**
- **Sibayan, Gian Eugene**
- **Umali, Ralph Dwayne**

---

### 🔗 Links 
-  [Streamlit App Link](https://group8-final-project.streamlit.app/)
-  [Google Colab Notebook](https://colab.research.google.com/drive/1wd2m0H3kK7kpx-FXZEfybRLnjRESx-rU?usp=sharing)

 ### 📊 Dataset
-  [Coffee Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset?select=coffee_analysis.csv)

---

### Project Pages and Features
1. `Dataset` – A preview of the Coffee Reviews Dataset, the dataset used for this project. 
2. `Data Pre-processing` – Processing the data by removing nulls, duplicates, and outliers, tokenizing text data, and encoding object columns.
3. `EDA` – An Exploratory Data Analysis that focuses on data distribution and correlation of variables tailored for the models we aim to develop. These are visualized through pie charts, word clouds, histograms, and bar charts.
4. `Coffee Price Prediction Model` – A model that uses Random Forest Regressor to predict coffee prices based on categorical data.
5. `Sentiment Intensity Analysis Model` – A model that uses Random Forest Regressor to predict ratings based on coffee descriptions.
6. `Coffee Recommendation Model` – A model that uses TF-IDF vectorizer and cosine similarity to curate coffee recommendations based on similar descriptions.
7. `Coffee Clustering Model` – A model that uses K-means to group coffees based on price and rating patterns.
8. `Conclusion` – Contains a summary of the processes and insights gained from the previous steps. 

---

## 💡 Findings / Insights  

1. **Dataset Characteristics**  
   - The most significant features for price prediction were bean origins, while the sentiment intensity analyzer and recommendation models primarily relied on descriptions. The clustering model used both rating and price to analyze patterns.

2. **Feature Distributions and Separability**  
   - The most significant features for price prediction were bean origins, while the sentiment intensity analyzer and recommendation models primarily relied on descriptions. The clustering model used both rating and price to analyze patterns.

3. **Model Performance (Price Prediction)**  
   - The price prediction model achieved high accuracy, demonstrating effective use of categorical and numerical features.  
   - Key predictors include origin and coffee type.  

4. **Model Performance (Sentiment Intensity Analysis)**  
   - The sentiment analysis model effectively translates coffee descriptions into predicted ratings, revealing clear sentiment trends in reviews.  

5. **Model Performance (Coffee Recommendation)**  
   - The recommendation model accurately identifies coffees with similar descriptions, proving useful for personalization.  

6. **Model Performance (Clustering Analysis)**  
   - The clustering model successfully grouped coffees into clusters with shared pricing and rating characteristics, uncovering market trends.  
   - The model was able split the coffees into 3 groups: low price with average rating, average price with high rating, and high price with average to high ratings.
  

---

### Summing up:

This project showed how the Coffee Reviews Dataset can be used to uncover insights and build reliable models. Its detailed structure made it easy to work with, requiring minimal cleaning. The models performed well, predicting prices, analyzing sentiment, and clustering coffees effectively. While there were challenges like variations in descriptions, the results highlight how data and machine learning can be combined to explore trends and preferences in the coffee market.

---
