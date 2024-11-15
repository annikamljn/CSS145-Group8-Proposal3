#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import string
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import gaussian_kde
from io import BytesIO
from sklearn.metrics import mean_squared_error, r2_score

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#######################
# Page configuration
st.set_page_config(
    page_title="Dashboard Template", # Replace this with your Project's Title
    page_icon="☕", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Dashboard Template')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"
        
    if st.button("Coffee Price Prediction Model", use_container_width=True, on_click=set_page_selection, args=('coffee_price_prediction',)):
        st.session_state.page_selection = "coffee_price_prediction"

    if st.button("Sentiment Intensity Analysis Model", use_container_width=True, on_click=set_page_selection, args=('sentiment_model',)):
        st.session_state.page_selection = "sentiment_model"

    if st.button("Coffee Recommendation Model", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "coffee_rec_model"

    if st.button("Coffee Clustering Model", use_container_width=True, on_click=set_page_selection, args=('clustering_analysis',)):
        st.session_state.page_selection = "clustering_analysis"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. BUNAG, Annika\n2. CHUA, Denrick Ronn\n3. MALLILLIN, Loragene\n4. SIBAYAN, Gian Eugene\n5. UMALI, Ralph Dwayne")

#######################
# Data

# Load data
dataset = pd.read_csv("data/coffee_analysis.csv")
df_initial = dataset

#######################

# Functions

def create_pie_chart(df, names_col, values_col, width, height, key, title):
    # Generate a pie chart
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        title=title
    )
    # Adjust the layout for the pie chart
    fig.update_layout(
        width=width,  
        height=height,  
        showlegend=True
    )
    # Display the pie chart in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=f"pie_chart_{key}")

def plot_price_distribution(df):
    # Create a histogram using Plotly
    fig = go.Figure()

    # Add histogram bars with outlines
    fig.add_trace(go.Histogram(
        x=df['100g_USD'],
        nbinsx=20,
        marker=dict(line=dict(color='black', width=1)),  # Outline color and width
        name='Price Distribution'
    ))

    # Calculate the KDE for the trendline
    kde = gaussian_kde(df['100g_USD'])
    x_range = np.linspace(df['100g_USD'].min(), df['100g_USD'].max(), 100)
    kde_values = kde(x_range)

    # Scale the KDE to match the histogram frequency
    bin_width = (df['100g_USD'].max() - df['100g_USD'].min()) / 20
    kde_scaled = kde_values * len(df) * bin_width

    # Add the KDE line to the histogram
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_scaled,
        mode='lines',
        name='Trendline',
        line=dict(color='blue', width=2)
    ))

    # Update layout for better visuals
    fig.update_layout(
        title='Distribution of Price per 100g',
        xaxis_title='Price per 100g (USD)',
        yaxis_title='Frequency'
    )

    return fig

#######################

# Pages

###################################################################
# About Page ######################################################
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.markdown("""
    This Streamlit dashboard explores a dataset about coffee, which includes information about its name, type, and origins, as well as descriptive reviews. To analyze the data, data preprocessing and Exploratory Data Analysis (EDA) are conducted to gain insights, which are then used to train Machine Learning Models to predict coffee prices, predict ratings based on descriptions, curate coffee recommendations, and group them based on similar characteristics.
    
    #### Pages

    1. `Dataset` – A preview of the Coffee Reviews Dataset, the dataset used for this project. 
    2. `Data Pre-processing` – Processing the data by removing nulls, duplicates, and outliers, tokenizing text data, and encoding object columns.
    3. `EDA` – An Exploratory Data Analysis that focuses on data distribution and correlation of variables tailored for the models we aim to develop. These are visualized through pie charts, word clouds, histograms, and bar charts.
    4. `Coffee Price Prediction Model` – A model that uses Random Forest Regressor to predict coffee prices based on categorical data.
    5. `Sentiment Intensity Analysis Model` – A model that uses Random Forest Regressor to predict ratings based on coffee descriptions.
    6. `Coffee Recommendation Model` – A model that uses TF-IDF vectorizer and cosine similarity to curate coffee recommendations based on similar descriptions.
    7. `Coffee Clustering Model` – A model that uses K-means to group coffees based on price and rating patterns.
    8. `Conclusion` – Contains a summary of the processes and insights gained from the previous steps. 
    """)

###################################################################
# Dataset Page ####################################################
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")
    
    st.write("The dataset by `@schmoyote` contains information about various coffee blends, their origins, roasters, and ratings. More specifically, the dataset displays details such as the name of the blend, its origins, the company that produced it, its price, and descriptive reviews. This dataset would be an invaluable tool to create a framework for predicting trends in coffee preferences, roaster locations, and blend ratings.")
    st.write("Additionally, this could help businesses improve their coffee selections on what can be altered and added to make their customers’ experience better and feel more personalized. Thanks to machine learning models, and data analysis this dataset will provide much more opportunities not only to coffee organizations but to their consumers as well.")

    st.markdown("**Content**")
    st.write("The dataset contains a total of 2095 rows containing unique entries of coffee and 12 columns, containing the following attributes: coffee name, roaster name, roast type, roaster location, bean origins, coffee price, coffee rating, review date, and coffee descriptions.")

    st.markdown("`Link: ` https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset")

    st.subheader("Displaying the dataset as a data frame")
    st.write(df_initial)
    st.subheader("Displaying the descriptive statistics")
    st.write(df_initial.describe())

    st.markdown("""
    The dataset only contains two numerical columns, namely `100g_USD` (price) and `rating`. Running the code df.describe() reveals that, for the price column, the standard deviation is 11.4, which means that the prices throughout the dataset are widely spread out. On the other hand, the rating’s standard deviation is 1.56, which means that the values are also spread out, but significantly less compared to the price. The mean of the dataset’s prices is approximately 9.32 USD, while the rating’s mean is around 9.11 out of 100. 
    
    The minimum value for the price is 0.12 USD, while the maximum value is 132.28. This is a significant jump in value, which may be caused by factors such as roaster location or bean origins.

    The minimum value for rating is 84, while the maximum value is 98. This suggests that the dataset’s contents primarily contain highly rated coffees. 
    """)
    

###################################################################
# Data Cleaning Page ##############################################
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Display the initial dataset
    st.subheader("Initial Dataset", divider=True)
    st.write(df_initial.head())

    # Making a copy before pre-processing
    df = df_initial.copy()

    ##### Removing nulls
    st.subheader("Removing nulls, duplicates, etc.", divider=True)
    st.markdown("In this step, we will identify and remove any missing values, as these can lead to inaccurate analyses. We’ll also check for duplicate rows to ensure the dataset’s integrity.")
    missing_values = round((df.isnull().sum()/df.shape[0])*100, 2)
    st.write(missing_values)

    # Code snippet for removing null values
    st.code("""
df = df.dropna()
    """, language="python")
    
    # Drop NA
    df = df.dropna()
    st.write("Missing values after dropping NA:")
    st.write(round((df.isnull().sum()/df.shape[0])*100, 2))

    # Check for duplicates
    duplicate_rows = df[df.duplicated()]
    num_duplicate_rows = len(duplicate_rows)
    st.write(f"Number of duplicate rows: {num_duplicate_rows}")

    ##### Removing Outliers
    st.subheader("Removing Outliers", divider=True)
    st.markdown("Outliers can significantly skew analysis results. We’ll use the Interquartile Range (IQR) method to identify and remove outliers in the 100g_USD (price per 100g) and rating columns.")
    plt.figure(figsize=(10, 4))
    plt.boxplot(df['100g_USD'], vert=False)
    plt.ylabel('Price Column (100g_USD)')
    plt.xlabel('Price Values')
    plt.title('Distribution of Coffee Price per 100g (USD)')
    st.pyplot(plt)

    Q1 = df['100g_USD'].quantile(0.25)
    Q3 = df['100g_USD'].quantile(0.75)
    IQR = Q3 - Q1
    price_lower_bound = max(0, Q1 - 1.5 * IQR)
    price_upper_bound = Q3 + 1.5 * IQR

    # Code snippet for removing outliers
    st.code("""
Q1 = df['100g_USD'].quantile(0.25)
Q3 = df['100g_USD'].quantile(0.75)
IQR = Q3 - Q1
price_lower_bound = max(0, Q1 - 1.5 * IQR)
price_upper_bound = Q3 + 1.5 * IQR

df = df[(df['100g_USD'] >= price_lower_bound) & (df['100g_USD'] <= price_upper_bound)]
    """, language="python")
    
    st.write('Lower Bound:', price_lower_bound)
    st.write('Upper Bound:', price_upper_bound)

    # Filter the dataset to remove outliers
    df = df[(df['100g_USD'] >= price_lower_bound) & (df['100g_USD'] <= price_upper_bound)]

    # Display price statistics
    st.markdown("**Price Statistics after Outlier Removal:**")
    st.write(df['100g_USD'].describe())

    # Rating Outliers
    st.subheader("Rating Outliers")
    st.markdown("We will also remove outliers from the rating column using the same IQR method.")
    plt.figure(figsize=(10, 4))
    plt.boxplot(df['rating'], vert=False)
    plt.ylabel('Rating Column')
    plt.xlabel('Rating Values')
    plt.title('Distribution of Coffee Ratings')
    st.pyplot(plt)

    Q1 = df['rating'].quantile(0.25)
    Q3 = df['rating'].quantile(0.75)
    IQR = Q3 - Q1
    rating_lower_bound = max(0, Q1 - 1.5 * IQR)
    rating_upper_bound = Q3 + 1.5 * IQR

    # Code snippet for rating outliers removal
    st.code("""
Q1 = df['rating'].quantile(0.25)
Q3 = df['rating'].quantile(0.75)
IQR = Q3 - Q1
rating_lower_bound = max(0, Q1 - 1.5 * IQR)
rating_upper_bound = Q3 + 1.5 * IQR

df = df[(df['rating'] >= rating_lower_bound) & (df['rating'] <= rating_upper_bound)]
    """, language="python")

    st.write('Lower Bound:', rating_lower_bound)
    st.write('Upper Bound:', rating_upper_bound)

    # Filter the dataset to remove outliers
    df = df[(df['rating'] >= rating_lower_bound) & (df['rating'] <= rating_upper_bound)]

    # Display rating statistics
    st.markdown("**Rating Statistics after Outlier Removal:**")
    st.write(df['rating'].describe())

    
    ##### Text pre-processing
    st.subheader("Coffee review text pre-processing", divider=True)
    st.markdown("Text data often needs to be cleaned and processed to extract insights. Here, we will perform the following preprocessing steps on the coffee review descriptions:\n"
                "- Convert text to lowercase.\n"
                "- Remove punctuation.\n"
                "- Tokenize words.\n"
                "- Remove stopwords (common words like 'the' and 'is' that don't add much meaning).\n"
                "- Lemmatize words (reduce words to their base form).")
    # Code snippet for text pre-processing
    st.code("""
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Stopword Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

df['desc_1_processed'] = df['desc_1'].apply(preprocess_text)
df['desc_2_processed'] = df['desc_2'].apply(preprocess_text)
df['desc_3_processed'] = df['desc_3'].apply(preprocess_text)
    """, language="python")


    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

   
    st.write(df[['desc_1', 'desc_2', 'desc_3']].head())

    # Combining all the preprocessing techniques in one function
    def preprocess_text(text):
        # 1. Lowercase
        text = text.lower()

        # 2. Remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 3. Tokenize
        tokens = word_tokenize(text)

        # 4. Stopword Removal
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # 5. Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens


    # Apply the preprocessing
    df['desc_1_processed'] = df['desc_1'].apply(preprocess_text)
    df['desc_2_processed'] = df['desc_2'].apply(preprocess_text)
    df['desc_3_processed'] = df['desc_3'].apply(preprocess_text)


    
    
    # Display the processed DataFrame
    st.write(df[['desc_1_processed', 'desc_1', 'desc_2_processed', 'desc_2', 'desc_3_processed', 'desc_3']].head())

    ##### Encoding object columns
    st.subheader("Encoding object columns", divider=True)
    st.markdown("Encoding categorical columns is essential to make them usable in machine learning models. Here, we use label encoding to convert text-based categorical values into numeric form.")

    # Display DataFrame info
    info_buffer = st.empty()  

    # Create a summary DataFrame for display
    info_data = {
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum(),
        "Dtype": df.dtypes
    }

    info_df = pd.DataFrame(info_data)

    st.dataframe(info_df)

    # Initialize the LabelEncoder
    encoder = LabelEncoder()

    # Encode object columns
    df['name_encoded'] = encoder.fit_transform(df['name'])
    df['roaster_encoded'] = encoder.fit_transform(df['roaster'])
    df['roast_encoded'] = encoder.fit_transform(df['roast'])
    df['loc_country_encoded'] = encoder.fit_transform(df['loc_country'])
    df['origin_1_encoded'] = encoder.fit_transform(df['origin_1'])
    df['origin_2_encoded'] = encoder.fit_transform(df['origin_2'])

    # Store the cleaned DataFrame in session state
    st.session_state.df = df

    # Display encoded columns
    st.subheader("Encoded Columns Preview")
    st.write(df[['name', 'name_encoded', 'roast', 'roast_encoded', 'roaster', 'roaster_encoded']].head())
    st.write(df[['loc_country', 'loc_country_encoded', 'origin_1', 'origin_1_encoded', 'origin_2', 'origin_2_encoded']].head())

    # Create and display summary mapping DataFrames
    def display_summary_mapping(column_name, encoded_column_name):
        unique_summary = df[column_name].unique()
        unique_summary_encoded = df[encoded_column_name].unique()
        summary_mapping_df = pd.DataFrame({column_name: unique_summary, f'{column_name}_encoded': unique_summary_encoded})
        st.write(f"{column_name} Summary Mapping")
        st.dataframe(summary_mapping_df)

    # Display mappings for each encoded column
    display_summary_mapping('name', 'name_encoded')
    display_summary_mapping('roast', 'roast_encoded')
    display_summary_mapping('roaster', 'roaster_encoded')
    display_summary_mapping('loc_country', 'loc_country_encoded')
    display_summary_mapping('origin_1', 'origin_1_encoded')
    display_summary_mapping('origin_2', 'origin_2_encoded')
    
###################################################################
# EDA Page ########################################################
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    # Check if the cleaned DataFrame exists in session state
    if 'df' in st.session_state:
        df = st.session_state.df

        col = st.columns((2, 3, 3), gap='medium')

        with col[0]:
            with st.expander('Legend', expanded=True):
                st.write('''
                    - Data: [Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset).
                    - :orange[**Pie Chart**]: Distribution of coffee types, roasters, roaster locations, and bean origins.
                    - :orange[**Word Cloud**]: Word frequencies in descriptive reviews.
                    - :orange[**Histogram**]: Distribution of coffee prices and ratings.
                    - :orange[**Bar Graph and Box Plot**]: Correlation analyses between categorical and numerical data.
                    ''')

            ##### Roast types chart #####
            roast_counts = df['roast'].value_counts().reset_index()
            roast_counts.columns = ['Roast', 'Count']
            create_pie_chart(roast_counts, 'Roast', 'Count', width=400, height=400, key='coffee_roast', title='Distribution of Roast Types')     

        with col[1]:
            ##### Roasters pie chart #####
            top_n = 20

            roaster_counts = df['roaster'].value_counts()
            top_roasters = roaster_counts.nlargest(top_n)
            other_roasters = roaster_counts.iloc[top_n:].sum()

            roaster_counts_top = pd.concat([top_roasters, pd.Series({'Other': other_roasters})]).reset_index()
            roaster_counts_top.columns = ['Roaster', 'Count']

            create_pie_chart(roaster_counts_top, 'Roaster', 'Count', width=400, height=400, key='top_roasters', title='Distribution of Top 20 Coffee Roasters')

            ##### Bean Origins 1 Pie Chart #####
            origin_counts = df['origin_1'].value_counts()
            top_origins = origin_counts.nlargest(top_n)
            other_origins = origin_counts.iloc[top_n:].sum()

            origin_counts_top = pd.concat([top_origins, pd.Series({'Other': other_origins})]).reset_index()
            origin_counts_top.columns = ['Origin', 'Count']

            create_pie_chart(origin_counts_top, 'Origin', 'Count', width=400, height=400, key='bean_origins', title=f'Distribution of Top {top_n} Bean Origins in "origin_1"')

        with col[2]:
            ##### Roaster Locations Pie Chart #####
            loc_counts = df['loc_country'].value_counts()

            loc_counts_df = loc_counts.reset_index()
            loc_counts_df.columns = ['Location', 'Count']
            create_pie_chart(loc_counts_df, 'Location', 'Count', width=400, height=400, key='roaster_locations', title='Distribution of Roaster Locations')

            ##### Bean Origins 2 Pie Chart #####
            origin_counts_2 = df['origin_2'].value_counts()
            top_origins_2 = origin_counts_2.nlargest(top_n)
            other_origins_2 = origin_counts_2.iloc[top_n:].sum()

            origin_counts_top_2 = pd.concat([top_origins_2, pd.Series({'Other': other_origins_2})]).reset_index()
            origin_counts_top_2.columns = ['Origin', 'Count']

            create_pie_chart(origin_counts_top_2, 'Origin', 'Count', width=400, height=400, key='bean_origins_2', title=f'Distribution of Top {top_n} Bean Origins in "origin_2"')
        
        st.subheader("Insights", divider=True)
        st.markdown("#### Data Distribution Analysis")

        st.markdown("##### ➡️ Categorical Data")
        create_pie_chart(roast_counts, 'Roast', 'Count', width=400, height=400, key='coffee_roast', title='Distribution of Roast Types')
        st.markdown("Based on the graph, we can infer that **Medium-Light roasts** are the **most prominent** types, spanning almost three-fourths (72.8%) of the dataset. Around **one-fourth** of the dataset consists of nearly equal distributions of **Light and Medium roasts**. **Medium-Dark and Dark** have the **least** number of instances, encompassing only less than 2% of the dataset when combined. It might be best to balance the dataset to lessen the bias towards Medium-Light coffees in model training.")

        create_pie_chart(roaster_counts_top, 'Roaster', 'Count', width=400, height=400, key='top_roasters', title='Distribution of Top 20 Coffee Roasters')
        st.markdown("Due to the numerous unique roaster names within the dataset, the graph was adjusted to highlight the top 20 most prominent roasters while grouping the rest as “others.” The **top 20 roasters** span **less than half** of the entire dataset, as other roasters below the top 20 take up 57.7%. **JBC Coffee Roasters** and **Kakalove Café** have the **most occurrences**, encompassing **8.7%** and **7.2%** of the dataset respectively.")

        create_pie_chart(loc_counts_df, 'Location', 'Count', width=400, height=400, key='roaster_locations', title='Distribution of Roaster Locations')
        st.markdown("The **majority** of the roasters in the dataset are based in the **United States**, spanning **66%**. The **second most prominent** roaster location is **Taiwan**, encompassing **27.1%** of the dataset. It is apparent from this graph that the data for roaster locations are largely imbalanced, so it is important to take note of this when training models that involve the use of this data column.")

        create_pie_chart(origin_counts_top, 'Origin', 'Count', width=400, height=400, key='bean_origins', title=f'Distribution of Top {top_n} Bean Origins in "origin_1"')
        st.markdown("Similar to the circumstances with the roasters, there are too many unique values for “origin_1” to coherently present the graph, thus it is presented with only the top 20 bean origins. **The top 20 origins** for this column only encompass less than half of the entire dataset, summing up to **only 38.7%**. The **most recurring** origins are **Guji Zone, Yirgacheffe Growing Region, Ethiopia, and Nyeri Growing Region**, which take up approximately **3-7% each**. The rest of the top regions only span around 2% or less each.")

        create_pie_chart(origin_counts_top_2, 'Origin', 'Count', width=400, height=400, key='bean_origins_2', title=f'Distribution of Top {top_n} Bean Origins in "origin_2"')
        st.markdown("For the “origin_2” column, the **top 20 bean origins** comprise **61.6%** of the entire dataset. The **most recurring** origins are **Southern Ethiopia, Colombia, Oromia Region, Guatemala, and South-Central Kenya**, **each encompassing around 6-7%** of the entire dataset. The rest of the top regions only appear to take up 4% or less of the dataset.")

        #########################################
        #########################################
        st.markdown("##### ➡️ Numerical Data")
        ##### Price Distribution #####
        price_distribution_fig = plot_price_distribution(df)
        st.plotly_chart(price_distribution_fig)
        st.markdown("The most **frequent found price range** in the dataset seems to be around **5 to 6 USD per 100 grams** while there are nearly **400 products** within it. Few of the websites offer products for a low price of almost **0 to 2 USD**, and few of the high prices are **around 14 USD**.")

        ##### Rating Distribution #####
        fig = go.Figure()

        # Add histogram bars with outlines
        fig.add_trace(go.Histogram(
            x=df['rating'],
            xbins=dict(start=df['rating'].min(), end=df['rating'].max() + 1, size=1), 
            marker=dict(line=dict(color='black', width=1)),  
            name='Rating Distribution'
        ))

        # Update layout for better visuals
        fig.update_layout(
            title='Distribution of Ratings',
            xaxis_title='Rating',
            yaxis_title='Frequency',
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        st.markdown("The **most common rating** of the coffees in this dataset can be between **93 to 94**. The lowest ratings go around **89 to 90** and the highest ratings insulate around **97 and 98**.")

        st.markdown("##### ➡️ Text Data")

        #### Word Distribution #####
        all_tokens = df['desc_1_processed'].sum() + df['desc_2_processed'].sum() + df['desc_3_processed'].sum()

        # Create a space-separated string for WordCloud
        text_for_wordcloud = ' '.join(all_tokens)

        # Generate the WordCloud
        wordcloud = WordCloud(
            width=600,  
            height=400,  
            background_color='white',
            max_words=200,
            colormap='viridis',
            stopwords=None  # We've already removed stopwords during preprocessing
        ).generate(text_for_wordcloud)

        # Create a BytesIO buffer to hold the image
        buffer = BytesIO()
        plt.figure(figsize=(6, 4))  # Adjust the figure size
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        # Save the figure to the buffer
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)  # Remove extra padding
        buffer.seek(0)  # Move to the beginning of the buffer

        # Display the Word Cloud in Streamlit
        st.image(buffer, caption='Word Cloud from Processed Descriptions', use_column_width=False, width=600)  # Set a specific width

        plt.close()
        st.markdown("The word cloud presents a summary of the most common words from columns “desc_1”, “desc_2”, and “desc_3” combined. The most prominent words across all the reviews are **“aroma,” “cup,” “variety,”** and **“arabica,”** followed by **“call,” “information,” “dark,” “chocolate,”** and lastly, **“processed.”** Observing the graph, the majority of the words that appear are mostly **objective** rather than subjective. The dataset owner indicates that these columns are reviews, however, it may appear that these lean more toward being descriptive text than sentiments.")

        st.divider()

        #########################################
        #########################################
        st.markdown("#### Correlation Analysis")

        #########################################
        #### Price Prediction Analysis EDA ###
        st.markdown("##### ➡️ For the Price Prediction Model")
        
        #### Price by Type ####
        fig = px.box(df, x='roast', y='100g_USD', title='Price per 100g by Roast Level',
                    labels={'roast': 'Roast', '100g_USD': 'Price per 100g (USD)'})

        # Update layout for better appearance
        fig.update_layout(xaxis_title='Roast', yaxis_title='Price per 100g (USD)', xaxis_tickangle=-45)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=False)  
        st.markdown("A look at the graph also reveals that **Medium-Dark roast** coffee's prices of the different types are spread out more compared to the others; prices range between **4 USD and 8.5 USD**. **Medium roasts** are the smallest part of the general price range as it is usually priced at **4.5 to 6 USD**.")

        # Price by Type
        fig = px.box(df, x='roast', y='100g_USD', title='Price per 100g by Roast Level',
                    labels={'roast': 'Roast', '100g_USD': 'Price per 100g (USD)'})

        # Update layout for better appearance
        fig.update_layout(xaxis_title='Roast', yaxis_title='Price per 100g (USD)', xaxis_tickangle=-45)

        # Display the Plotly chart
        st.plotly_chart(fig, use_container_width=True)  # Use container width for the Plotly chart

        st.markdown("A look at the graph also reveals that **Medium-Dark roast** coffee's prices of the different types are spread out more compared to the others; prices range between **4 USD and 8.5 USD**. **Medium roasts** are the smallest part of the general price range as it is usually priced at **4.5 to 6 USD**.")

        #### Price by Rating and Type ####
        st.markdown("**Average Price per 100g by Rating Bins and Roast Type**")
        # Define bin edges
        bin_edges = [89, 91, 93, 95, 97]

        # Create bins based on the custom edges
        df['rating_bins'] = pd.cut(df['rating'], bins=bin_edges)

        # Convert the 'rating_bins' to string for better labeling
        df['rating_bins'] = df['rating_bins'].astype(str)

        # Calculate the average price per bin
        avg_price_per_bin = df.groupby(['rating_bins', 'roast'])['100g_USD'].mean().reset_index()

        # Create columns for the second graph
        col1, col2 = st.columns(2)

        # Plotting using Matplotlib and Seaborn with fixed width
        with col1:
            plt.figure(figsize=(8, 5))  # Set fixed width and height
            sns.barplot(data=avg_price_per_bin, x='rating_bins', y='100g_USD', hue='roast', palette='viridis')
            plt.title('Average Price per 100g by Rating Bins and Roast Type')
            plt.xlabel('Rating Bins')
            plt.ylabel('Average Price per 100g (USD)')
            plt.xticks(rotation=45)
            plt.legend(title='Roast', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(plt)

        with col2:
            st.markdown("The graph provided shows us that, generally, coffees with **higher ratings** tend to have a **higher average price**. Dark, Light, Medium, and Medium-dark coffees tend to increase in average price as their ratings increase. On the other hand, across all rating ranges, **Medium-light coffees** tend to maintain the **same price range**.")

        #### Price by Country ####
        # Calculate the average price per country
        avg_price_by_country = df.groupby('loc_country')['100g_USD'].mean().reset_index()

        # Sort values by average price
        avg_price_by_country = avg_price_by_country.sort_values(by='100g_USD')

        # Plotting using Plotly
        fig = px.bar(avg_price_by_country, 
                    x='100g_USD', 
                    y='loc_country', 
                    orientation='h', 
                    title='Average Price per 100g by Roaster Country',
                    labels={'100g_USD': 'Average Price per 100g (USD)', 'loc_country': 'Country'},
                    color='100g_USD',
                    color_continuous_scale='Teal')

        # Update layout for better appearance
        fig.update_layout(xaxis_title='Average Price per 100g (USD)', yaxis_title='Country')

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        st.markdown("The countries with the **highest average price** per 100 grams of coffee are **Honduras, Hawai’i, Japan, Hong Kong, and New Taiwan**, with an estimated 10 to 12 USD range. Countries with the **cheapest coffee** prices are **Peru, Uganda, Guatemala, China, and Belgium**, with prices averaging 4 USD or less. Other countries present in the dataset price their coffees around 5 to 8 USD.")

        #### Price by origin ####
        # Set the number of top origins to display
        top_n = 10

        # Identify the top N origins in 'origin_2'
        top_origins = df['origin_2'].value_counts().nlargest(top_n).index

        # Create a new column where non-top origins are labeled as 'Others'
        df['origin_grouped'] = df['origin_2'].apply(lambda x: x if x in top_origins else 'Others')

        # Create a box plot for price by grouped origin using Plotly
        fig = px.box(df, 
                    x='origin_grouped', 
                    y='100g_USD', 
                    category_orders={'origin_grouped': top_origins.tolist() + ['Others']},
                    title=f'Price per 100g by Top {top_n} Bean Origins (origin_2), Others Grouped',
                    labels={'origin_grouped': 'Individual Origin', '100g_USD': 'Price per 100g (USD)'})

        # Update layout for better appearance
        fig.update_layout(xaxis_title='Individual Origin', yaxis_title='Price per 100g (USD)', xaxis_tickangle=-45)

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        st.markdown("Present in the graph are average prices with respect to the top ten most common bean origins from the “origin_2” column. Among these, **Colombia** generally has the **most expensive** coffee prices per 100 grams, as well as the **widest price range**, with the majority ranging from **5 to 9 USD**. **Guatemala, Ethiopia, and Gedeo Zone** appear to have the **cheapest coffees**, which range from **5 to 6 USD**.")

        #########################################
        #### Sentiment Intensity Analysis EDA ###
        st.divider()
        st.markdown("##### ➡️ For the Sentiment Intensity Analysis Model")
        # Combine the description columns into a single processed text column
        df['combined_desc'] = (
            df['desc_1_processed'].fillna('').apply(' '.join) + ' ' +
            df['desc_2_processed'].fillna('').apply(' '.join) + ' ' +
            df['desc_3_processed'].fillna('').apply(' '.join)
        )

        # Function to create a word cloud based on a range of ratings
        def create_wordcloud(data, rating_min, rating_max):
            # Filter the dataset based on the rating range
            filtered_data = data[(data['rating'] >= rating_min) & (data['rating'] <= rating_max)]
            
            # Combine the text data from the 'combined_desc' column
            text = ' '.join(filtered_data['combined_desc'].dropna())

            # Generate the word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                collocations=False,
                stopwords=None
            ).generate(text)

            return wordcloud

        # Create columns for displaying word clouds
        col1, col2, col3 = st.columns(3)

        # Descriptions for each quartile
        quartile_1_desc = """
        **Quartile 1 (25%)**
        Descriptions in this range focus on broad qualities like **"mouthfeel," "sweet," "chocolate," “aroma,” and "finish,"** suggesting that these coffees are enjoyable but lack detailed complexity.
        """

        quartile_2_desc = """
        **Quartile 2 (50%)**
        As ratings increase, words such as **"fruit"** and **“note”** appear more frequently, indicating that higher-rated coffees are associated with more specific flavor profiles.
        """

        quartile_3_desc = """
        **Quartile 3 (75%)**
        Terms like **"aroma," "finish," and "fruit"** appear at the highest scores, indicating a greater attention to the complexities of flavor and aftertaste.
        """

        # Display word cloud for Quartile 1 in the first column
        with col1:
            wordcloud_q1 = create_wordcloud(df, rating_min=89, rating_max=92)
            plt.figure(figsize=(6, 4))
            plt.imshow(wordcloud_q1, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            st.write(quartile_1_desc)

        # Display word cloud for Quartile 2 in the second column
        with col2:
            wordcloud_q2 = create_wordcloud(df, rating_min=93, rating_max=93)
            plt.figure(figsize=(6, 4))
            plt.imshow(wordcloud_q2, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            st.write(quartile_2_desc)

        # Display word cloud for Quartile 3 in the third column
        with col3:
            wordcloud_q3 = create_wordcloud(df, rating_min=94, rating_max=97)
            plt.figure(figsize=(6, 4))
            plt.imshow(wordcloud_q3, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            st.write(quartile_3_desc)

        #########################################
        #### Recommendation Analysis EDA ###
        st.divider()
        st.markdown("##### ➡️ For the Coffee Recommendation Model")
        def preprocess_text(text):
            # Remove punctuation, convert to lowercase, and tokenize
            text = re.sub(r'[^\w\s]', '', text)
            text = text.lower()
            return text

        # Apply preprocessing to the 'name' column
        df['processed_name'] = df['name'].apply(preprocess_text)

        # Combine the processed description columns into a single text
        df['combined_desc'] = (
            df['desc_1_processed'].fillna('').apply(' '.join) + ' ' +
            df['desc_2_processed'].fillna('').apply(' '.join) + ' ' +
            df['desc_3_processed'].fillna('').apply(' '.join)
        )

        # Vectorize the 'processed_name' column
        vectorizer_name = CountVectorizer(max_features=100)
        X_name = vectorizer_name.fit_transform(df['processed_name']).toarray()

        # Vectorize the 'combined_desc' column
        vectorizer_desc = CountVectorizer(max_features=100)
        X_desc = vectorizer_desc.fit_transform(df['combined_desc']).toarray()

        # Convert to DataFrames for easier manipulation
        df_name_words = pd.DataFrame(X_name, columns=vectorizer_name.get_feature_names_out())
        df_desc_words = pd.DataFrame(X_desc, columns=vectorizer_desc.get_feature_names_out())

        # Align the features for correlation (intersect columns to avoid mismatch)
        common_words = list(set(df_name_words.columns) & set(df_desc_words.columns))

        # Use the list of common words to filter the DataFrames
        df_name_words = df_name_words[common_words]
        df_desc_words = df_desc_words[common_words]

        # Calculate correlation
        correlation_matrix = df_name_words.corrwith(df_desc_words, axis=0)

        # Create a DataFrame of correlations
        correlation_df = correlation_matrix.reset_index()
        correlation_df.columns = ['Word', 'Correlation']

        # Get the top 20 most correlated words
        top_correlated = correlation_df.sort_values(by='Correlation', key=abs, ascending=False).head(20)

        # Plotting with Plotly
        fig = px.bar(top_correlated, x='Correlation', y='Word', 
                    title='Top Correlations Between Words in "name" and "combined_desc"',
                    labels={'Correlation': 'Correlation', 'Word': 'Word'},
                    orientation='h')
        st.plotly_chart(fig)
        st.markdown("The graph above depicts the degree of correlation between common words found in coffee names and combined descriptions. The words with the highest degree of correlation are **“kenya,” “espresso,” “ethiopia,” “blend,” “natural,”** and **“honey.”** with a correlation level around **0.6 or higher.** This represents how coffee names could be associated with their given description, which can help in identifying similarities for a coffee recommendation model.")

        #########################################
        #### Recommendation Analysis EDA ###
        st.divider()
        st.markdown("##### ➡️ For the Coffee Clustering Model")
        # Box plot for price distribution across ratings using Plotly
        fig = px.box(df, x='rating', y='100g_USD', title='100g Price Distribution across Ratings (Box Plot)',
                    labels={'rating': 'Rating', '100g_USD': '100g Price (USD)'})

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Add text explanation
        st.write("""
        Presented is another graph that compares coffee ratings and prices, but this time with individual x-ticks per rating. 
        As observed in this graph and the one earlier, the **prices** of coffees are generally **higher** when they have a **higher rating**. 
        Coffees **rated 95-96** have the **widest price ranges** in the dataset, and coffees **rated 97** have the **smallest price range**. 
        Overall, coffees **rated 94-96** tend to be the **most expensive**, while coffees **rated 89** are the **cheapest**. 
        The positive correlation between the price and rating can be utilized to create a clustering algorithm to group the coffees based on these characteristics.
        """)

    else:
        st.error("DataFrame not found. Please go to the Data Cleaning page first. (It is required that you scroll to the bottom)")

###################################################################
# Clustering Analysis ###########################################
elif st.session_state.page_selection == "clustering_analysis":
    st.header("🤖 Coffee Clustering Model")

    if 'df' not in st.session_state:
        st.error("Please process the data in the Data Cleaning page first")
        st.stop()

    df = st.session_state.df 

    tab1, tab2 = st.tabs(["Machine Learning", "Cluster Prediction"])

    with tab1:
    # Dropping unnecessary column and handling missing values
        df_clustering = df.drop(columns=['origin_1'], errors='ignore')
        df_clustering = df_clustering[['100g_USD', 'rating']].dropna()

        # Removing outliers based on 99th percentile
        price_quantile = df_clustering['100g_USD'].quantile(0.99)
        df_clustering = df_clustering[df_clustering['100g_USD'] < price_quantile]

        # Splitting data into training and test sets
        train_data, test_data = train_test_split(df_clustering, test_size=0.3, random_state=42)

        # Standardizing the data
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Finding optimal clusters using the elbow method
        inertia = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(train_scaled)
            inertia.append(kmeans.inertia_)

        # Plotting the Elbow curve using Plotly
        st.subheader('Elbow Method for Optimal Number of Clusters')
        elbow_fig = px.line(x=K, y=inertia, markers=True, labels={'x': 'Number of clusters', 'y': 'Inertia'})
        elbow_fig.update_layout(title='Elbow Method for Optimal Number of Clusters')
        st.plotly_chart(elbow_fig)  # Use Streamlit to display the plot

        # Choosing the optimal number of clusters (e.g., 3)
        optimal_clusters = 3
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        train_labels = kmeans.fit_predict(train_scaled)

        # Applying the model to the test data
        test_labels = kmeans.predict(test_scaled)

        # Adding cluster labels to the dataframes
        train_data['Cluster'] = train_labels
        test_data['Cluster'] = test_labels

        # Displaying Silhouette Score
        sil_score = silhouette_score(test_scaled, test_labels)
        st.write(f'Silhouette Score for {optimal_clusters} clusters on test data: {sil_score}')

        # Listing some data points from each cluster in the training set
        for cluster in range(optimal_clusters):
            st.write(f"\nSample data from Cluster {cluster}:")
            st.write(train_data[train_data['Cluster'] == cluster].head(5))

        # Visualizing the clusters for training data using Plotly
        st.subheader('Training Data Clusters Visualization')
        train_df = pd.DataFrame(train_scaled, columns=['Price per 100g (Scaled)', 'Rating (Scaled)'])
        train_df['Cluster'] = train_labels

        train_scatter_fig = px.scatter(train_df, x='Price per 100g (Scaled)', y='Rating (Scaled)', color='Cluster', title='Training Data Clusters')
        st.plotly_chart(train_scatter_fig)  # Use Streamlit to display the plot

        # Visualizing the clusters for test data using Plotly
        st.subheader('Test Data Clusters Visualization')
        test_df = pd.DataFrame(test_scaled, columns=['Price per 100g (Scaled)', 'Rating (Scaled)'])
        test_df['Cluster'] = test_labels

        test_scatter_fig = px.scatter(test_df, x='Price per 100g (Scaled)', y='Rating (Scaled)', color='Cluster', title='Test Data Clusters')
        st.plotly_chart(test_scatter_fig)  # Use Streamlit to display the plot

    with tab2:
    # User input for new price and rating
        st.subheader('Input Your Own Values for Prediction')
        user_price = st.number_input('Enter Price per 100g (USD):', min_value=0.0, format="%.2f", value=9.5)  # Default value set to 10.0
        user_rating = st.number_input('Enter Rating:', min_value=0.0, max_value=100.0, format="%.1f", value=92.0)  # Default value set to 85.0

        # Predicting the cluster for user input
        if st.button('Predict Cluster'):
            user_data = pd.DataFrame({'100g_USD': [user_price], 'rating': [user_rating]})
            user_scaled = scaler.transform(user_data)
            user_cluster = kmeans.predict(user_scaled)
            st.write(f'The predicted cluster for the input values is: {user_cluster[0]}')

            # Visualizing the user input data point
            user_df = pd.DataFrame({'Price per 100g (Scaled)': user_scaled[0][0], 'Rating (Scaled)': user_scaled[0][1], 'Predicted_Cluster': user_cluster[0]}, index=[0])
            comparison_fig = px.scatter(train_df, x='Price per 100g (Scaled)', y='Rating (Scaled)', color='Cluster', title='Comparison of User Input with Training Data')
            comparison_fig.add_scatter(x=user_df['Price per 100g (Scaled)'], y=user_df['Rating (Scaled)'], mode='markers', marker=dict(size=10, color='red'), name='User  Input', showlegend=True)
            st.plotly_chart(comparison_fig)  # Use Streamlit to display the plot

###################################################################
# Coffee Price Prediction Page ################################################
elif st.session_state.page_selection == "coffee_price_prediction":
    st.header("☕ Coffee Price Prediction Model")

    # Ensure DataFrame is available in session state
    if 'df' not in st.session_state:
        st.error("Please process the data in the Data Cleaning page first")
        st.stop()
    
    # Load the cleaned DataFrame
    df = st.session_state.df 

    # Set up tabs for the regression model
    tab1, tab2 = st.tabs(["Machine Learning", "Model Evaluation"])

    ### Tab 1: Machine Learning ###
with tab1:
    st.subheader("Model Training")
    
    # 1. Create a copy of the DataFrame for regression analysis
    df_coffeeprice_regression = df.copy()

    # Check if the necessary columns exist
    required_columns = ['roast', 'origin_2', 'loc_country', '100g_USD']
    if not all(col in df_coffeeprice_regression.columns for col in required_columns):
        st.error("The required columns are missing in the DataFrame.")
        st.stop()

    # 2. Select relevant columns and rename them if pre-processed
    df_coffeeprice_regression = df_coffeeprice_regression[required_columns]
    df_coffeeprice_regression = df_coffeeprice_regression.rename(columns={
        'origin_2': 'origin_2_processed',
        'loc_country': 'loc_processed'
    })

    # 3. Drop rows with missing values
    df_coffeeprice_regression = df_coffeeprice_regression.dropna()

    ### Outlier Handling ###
    # 4. Remove outliers using the Interquartile Range (IQR) method
    Q1 = df_coffeeprice_regression['100g_USD'].quantile(0.25)
    Q3 = df_coffeeprice_regression['100g_USD'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_coffeeprice_regression = df_coffeeprice_regression[
        (df_coffeeprice_regression['100g_USD'] >= lower_bound) & 
        (df_coffeeprice_regression['100g_USD'] <= upper_bound)
    ]

    ### Encoding Categorical Variables ###
    le_roast = LabelEncoder()
    le_origin_2 = LabelEncoder()
    le_location = LabelEncoder()

    df_coffeeprice_regression['roast'] = le_roast.fit_transform(df_coffeeprice_regression['roast'])
    df_coffeeprice_regression['origin_2_processed'] = le_origin_2.fit_transform(df_coffeeprice_regression['origin_2_processed'])
    df_coffeeprice_regression['loc_processed'] = le_location.fit_transform(df_coffeeprice_regression['loc_processed'])

    # 6. Define features (X) and target (y)
    X = df_coffeeprice_regression[['roast', 'origin_2_processed', 'loc_processed']]
    y = df_coffeeprice_regression['100g_USD']

    ### Model Training ###
    if st.button("Train Models"):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        dt_model = DecisionTreeRegressor(random_state=42)
        dt_model.fit(X_train, y_train)

        # Store models in session state
        st.session_state.rf_model = rf_model
        st.session_state.dt_model = dt_model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        st.success("Models trained successfully!")


    ### Tab 2: Model Evaluation ###
    with tab2:
        st.subheader("Model Evaluation")
        
        # 10. Evaluate Random Forest model performance
        rf_pred = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        st.write(f"Random Forest - Mean Squared Error (MSE): {rf_mse:.2f}")
        st.write(f"Random Forest - R² Score: {rf_r2:.2f}")

        # 11. Evaluate Decision Tree model performance
        dt_pred = dt_model.predict(X_test)
        dt_mse = mean_squared_error(y_test, dt_pred)
        dt_r2 = r2_score(y_test, dt_pred)
        st.write(f"Decision Tree - Mean Squared Error (MSE): {dt_mse:.2f}")
        st.write(f"Decision Tree - R² Score: {dt_r2:.2f}")

        ### Visualization: Actual vs Predicted Prices ###
        st.subheader("Actual vs Predicted Prices (Random Forest)")
        fig, ax = plt.subplots()
        ax.scatter(y_test, rf_pred, label='Random Forest', alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Actual Price (USD)')
        ax.set_ylabel('Predicted Price (USD)')
        ax.legend()
        st.pyplot(fig)

        ### Feature Importance ###
        st.subheader("Feature Importance (Random Forest)")
        rf_importance = rf_model.feature_importances_
        for i, score in enumerate(rf_importance):
            st.write(f"Feature: {X.columns[i]}, Score: {score:.4f}")

        ### User Input for Prediction ###
        st.subheader("Predict Coffee Price with Your Inputs")
        user_roast = st.selectbox('Select Roast Type:', le_roast.classes_)
        user_origin = st.selectbox('Select Bean Origin:', le_origin_2.classes_)
        user_location = st.selectbox('Select Roaster Location:', le_location.classes_)

        # Encode user inputs
        user_roast_encoded = le_roast.transform([user_roast])[0]
        user_origin_encoded = le_origin_2.transform([user_origin])[0]
        user_location_encoded = le_location.transform([user_location])[0]

        # Predict using Random Forest model
        user_input = pd.DataFrame({
            'roast': [user_roast_encoded], 
            'origin_2_processed': [user_origin_encoded], 
            'loc_processed': [user_location_encoded]
        })
        
        if st.button('Predict Price'):
            user_prediction = rf_model.predict(user_input)
            st.write(f"Predicted Coffee Price per 100g (USD): ${user_prediction[0]:.2f}")



                
###################################################################
# Description to Rating Page ################################################
elif st.session_state.page_selection == "sentiment_model":
    st.header("📊 Sentiment Intensity Analysis Model")

    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Define function to extract sentiment
    def extract_sentiment(text):
        sentiment_score = sia.polarity_scores(text)
        return sentiment_score['compound']  # Compound score as overall sentiment

    # Access the cleaned data from session state
    if 'df' not in st.session_state:
        st.error("Please process the data in the Data Cleaning page first")
        st.stop()

    df = st.session_state.df  # Use cleaned DataFrame stored in session state

    # Add tabs for different features
    tab1, tab2 = st.tabs(["Machine Learning", "Describe to Rate The Coffee"])

    with tab2:
        # Check if sentiment score columns exist; if not, create them
        if not all(col in df.columns for col in ['sentiment_score_1', 'sentiment_score_2', 'sentiment_score_3']):
            # Create sentiment score columns based on existing descriptions
            df['sentiment_score_1'] = df['desc_1'].apply(extract_sentiment)
            df['sentiment_score_2'] = df['desc_2'].apply(extract_sentiment)
            df['sentiment_score_3'] = df['desc_3'].apply(extract_sentiment)

        # Feature columns for training (using the three individual sentiment scores)
        X = df[['sentiment_score_1', 'sentiment_score_2', 'sentiment_score_3']]
        y = df['rating']

        # Split the data into training (70%) and testing (30%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the Random Forest Regressor
        rf = RandomForestRegressor(random_state=42)

        # Fit the model
        rf.fit(X_train, y_train)

        # Input field for coffee name
        coffee_names = df['name'].unique() 
        selected_coffee = st.selectbox("Select Coffee Name", coffee_names)

        # Get the existing descriptions for the selected coffee
        existing_desc = df[df['name'] == selected_coffee].iloc[0]
        desc_1 = existing_desc['desc_1']
        desc_2 = existing_desc['desc_2']
        desc_3 = existing_desc['desc_3']

        # Input fields for new coffee descriptions
        st.subheader("Describe This Coffee")
        new_desc_1 = st.text_area("New Description 1", "")
        new_desc_2 = st.text_area("New Description 2", "")
        new_desc_3 = st.text_area("New Description 3", "")

        if st.button("Predict Rating"):
            # Calculate sentiment scores for the new descriptions
            sentiment_score_1 = extract_sentiment(new_desc_1)
            sentiment_score_2 = extract_sentiment(new_desc_2)
            sentiment_score_3 = extract_sentiment(new_desc_3)

            # Prepare the input for prediction
            new_data = pd.DataFrame([[sentiment_score_1, sentiment_score_2, sentiment_score_3]], 
                                    columns=['sentiment_score_1', 'sentiment_score_2', 'sentiment_score_3'])

            # Predict the new rating
            predicted_rating = rf.predict(new_data)

            # Display the sentiment scores and predicted rating
            st.write("Sentiment Scores:")
            st.write(f"Description 1 Sentiment Score: {sentiment_score_1:.2f}")
            st.write(f"Description 2 Sentiment Score: {sentiment_score_2:.2f}")
            st.write(f"Description 3 Sentiment Score: {sentiment_score_3:.2f}")
            st.success(f"Predicted Rating for the provided descriptions: {predicted_rating[0]:.2f}")

    with tab1:
        st.subheader("Feature Importance Analysis")
        
        # Calculate the average sentiment score for each coffee description (mean of the 3 sentiment scores)
        df['average_sentiment'] = df[['sentiment_score_1', 'sentiment_score_2', 'sentiment_score_3']].mean(axis=1)

        # Rescale sentiment scores from range [0, 1] to range [84, 98]
        min_rating = 84
        max_rating = 98
        df['rescaled_sentiment'] = min_rating + (df['average_sentiment'] * (max_rating - min_rating))

        # Create hexbin plot using Plotly
        hexbin_fig = go.Figure(data=go.Histogram2d(
            x=df['rescaled_sentiment'],
            y=df['rating'],
            colorscale='Blues',
            colorbar=dict(title='Frequency'),
            histfunc='count',
        ))
        
        # Update layout for better visualization
        hexbin_fig.update_layout(
            title='Hexbin Plot of Sentiment Scores vs Ratings',
            xaxis_title='Sentiment Scores',
            yaxis_title='Ratings',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # Show the plot in Streamlit
        st.plotly_chart(hexbin_fig, use_container_width=True)

        st.write("""
        The hexbin plot shows the relationship between "Sentiment Scores" and "Ratings." The plot suggests a positive correlation between "Rescaled Sentiment Scores" and "Ratings." As the sentiment scores increase, the ratings tend to also increase.
        """)

        # Prepare data for scatter plot
        X = df[['sentiment_score_1', 'sentiment_score_2', 'sentiment_score_3']]
        y = df['rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and fit the Random Forest Regressor
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)

        # Create scatter plot using Plotly Express
        scatter_fig = px.scatter(
            df,
            x='rescaled_sentiment',
            y='rating',
            title='Scatter Plot of Rescaled Sentiment Scores vs Ratings',
            labels={'rescaled_sentiment': 'Rescaled Sentiment Scores', 'rating': 'Ratings'},
            color='rating',  # Optional: color by rating
            hover_data=['rescaled_sentiment', 'rating']  # Optional: show data on hover
        )
        
        # Update layout for better visualization
        scatter_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        # Show the plot in Streamlit
        st.plotly_chart(scatter_fig, use_container_width=True)

        # Calculate and display MAE and RMSE
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        st.write(f"The model's Mean Absolute Error (MAE) is 1.1728734991620522 and the Root Mean Squared Error (RMSE) is 1.5956235031050126. These metrics indicate the average magnitude of the model's prediction errors. A lower MAE suggests smaller average errors, while a lower RMSE suggests smaller average squared errors, with more weight given to larger errors.")

        st.markdown("""
        ### Understanding the Rating Prediction Model
        
        This rating prediction model utilizes multiple features derived from coffee descriptions to estimate the overall rating:
        
        1. **Description Sentiment Analysis**: Analyzes the sentiment of the coffee descriptions using Natural Language Processing (NLP) techniques to quantify the emotional tone and context.
        
        2. **Sentiment Scores**: Incorporates individual sentiment scores from multiple descriptions to capture different perspectives on the coffee's characteristics.
        
        3. **Model Training**: Utilizes a Random Forest Regressor to learn from historical data, identifying patterns and relationships between sentiment scores and actual ratings.
        
        The bar chart above represents the contribution of each sentiment score to the rating prediction.
        """)

###################################################################
# Prediction Page #################################################
elif st.session_state.page_selection == "coffee_rec_model":
    st.header("☕ Coffee Recommendation Model")
    
    # Check if the cleaned DataFrame exists in session state
    if 'df' not in st.session_state:
        st.error("Please process the data in the Data Cleaning page first")
        st.stop()
        
    df = st.session_state.df
    
    # Add tabs for different features
    tab1, tab2 = st.tabs(["Machine Learning", "Get Recommendations"])
    
    with tab2:
        # Create sidebar for filters
        with st.sidebar:
            st.subheader("Filter Options")
            
            # Price range filter
            price_range = st.slider(
                "Price Range (USD/100g)",
                float(df['100g_USD'].min()),
                float(df['100g_USD'].max()),
                (float(df['100g_USD'].min()), float(df['100g_USD'].max()))
            )
            
            # Roast type filter
            roast_types = ['All'] + list(df['roast'].unique())
            selected_roast = st.selectbox("Roast Type", roast_types)
            
            # Origin filter
            origins = ['All'] + list(df['origin_2'].unique())
            selected_origin = st.selectbox("Origin", origins)

        # Main content area
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Select Your Coffee")
            
            # Filter dataset based on selections
            filtered_df = df.copy()
            if selected_roast != 'All':
                filtered_df = filtered_df[filtered_df['roast'] == selected_roast]
            if selected_origin != 'All':
                filtered_df = filtered_df[filtered_df['origin_2'] == selected_origin]
            filtered_df = filtered_df[
                (filtered_df['100g_USD'] >= price_range[0]) &
                (filtered_df['100g_USD'] <= price_range[1])
            ]
            
            # Coffee selection
            selected_coffee = st.selectbox(
                "Choose a coffee to get recommendations",
                options=filtered_df['name'].unique()
            )

            if selected_coffee:
                coffee_info = filtered_df[filtered_df['name'] == selected_coffee].iloc[0]
                st.write("**Selected Coffee Details:**")
                st.write(f"🫘 Roast: {coffee_info['roast']}")
                st.write(f"📍 Origin: {coffee_info['origin_2']}")
                st.write(f"💰 Price: ${coffee_info['100g_USD']:.2f}/100g")
                st.write(f"⭐ Rating: {coffee_info['rating']}")

        with col2:
            if selected_coffee:
                st.subheader("Recommended Coffees")
                
                # Create recommendation dataframe
                df_reco = filtered_df.copy()
                columns_of_interest = ['name', 'roast', 'origin_1', 'origin_2', '100g_USD', 
                                     'desc_1_processed', 'desc_2_processed', 'desc_3_processed']
                df_reco = df_reco[columns_of_interest]

                # Combine descriptions
                df_reco['combined_desc'] = df_reco.apply(
                    lambda row: ' '.join(str(x) for x in [
                        row['desc_1_processed'],
                        row['desc_2_processed'],
                        row['desc_3_processed']
                    ]), 
                    axis=1
                )

                # Create TF-IDF matrix
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(df_reco['combined_desc'])
                
                # Calculate cosine similarity
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

                # Get recommendations
                idx = df_reco[df_reco['name'] == selected_coffee].index[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:6]  # Get top 5 similar coffees
                
                # Create recommendations with details
                for i, (idx, score) in enumerate(sim_scores, 1):
                    coffee = df_reco.iloc[idx]
                    with st.container():
                        st.markdown(
                            f"""
                            <div style='padding: 10px; border-radius: 5px; background-color: rgba(255, 255, 255, 0.05); margin-bottom: 10px;'>
                                <h3>{i}. {coffee['name']}</h3>
                                <p><strong>Similarity Score:</strong> {score:.2%}</p>
                                <p>🫘 <strong>Roast:</strong> {coffee['roast']}</p>
                                <p>📍 <strong>Origin:</strong> {coffee['origin_2']}</p>
                                <p>💰 <strong>Price:</strong> ${coffee['100g_USD']:.2f}/100g</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

    with tab1:
        st.subheader("Feature Importance Analysis")
        
        # Calculate and display feature importance
        def calculate_permutation_importance(df, baseline_sim_matrix, feature_columns):
            importance_scores = {}
            for feature in feature_columns:
                df_permuted = df.copy()
                df_permuted[feature] = np.random.permutation(df_permuted[feature].values)
                
                if feature in ['desc_1_processed', 'desc_2_processed', 'desc_3_processed']:
                    df_permuted['combined_desc'] = df_permuted.apply(
                        lambda row: ' '.join(str(x) for x in [
                            row['desc_1_processed'],
                            row['desc_2_processed'],
                            row['desc_3_processed']
                        ]), 
                        axis=1
                    )
                    tfidf_matrix_permuted = tfidf.fit_transform(df_permuted['combined_desc'])
                    permuted_sim_matrix = cosine_similarity(tfidf_matrix_permuted, tfidf_matrix_permuted)
                else:
                    permuted_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

                sim_diff = np.abs(baseline_sim_matrix - permuted_sim_matrix).mean()
                importance_scores[feature] = sim_diff
            
            return sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

        feature_columns = columns_of_interest
        importance_scores = calculate_permutation_importance(df_reco, cosine_sim, feature_columns)
        
        # Create and display bar chart
        importance_df = pd.DataFrame(importance_scores, columns=['Feature', 'Importance'])
        fig = px.bar(
            importance_df, 
            x='Feature', 
            y='Importance',
            title='Feature Importance in Coffee Recommendations',
            labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
        )
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Importance Score",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Training the TF-IDF model
        st.header("Training the TF-IDF Model")
        st.code("""
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df_reco['combined_desc'])
        """, language='python')
        st.write("TF-IDF model trained on `combined_desc` column.")

        
        st.header("Calculating Cosine Similarity")
        st.code("""
        # Calculating cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        """, language='python')
        st.write("Cosine similarity matrix calculated.")

        # Evaluation (since it's content-based, we skip traditional accuracy)
        st.header("Model Evaluation")
        st.write("In content-based recommendation systems, traditional accuracy metrics aren't typically used. Instead, we can consider relevance feedback, user ratings, or precision-at-k metrics for evaluation.")

        # Add explanation
        st.markdown("""
        ### Understanding the Model
        
        This recommendation system uses several features to find similar coffees:
        
        1. **Description Analysis**: Processes the coffee descriptions using NLP techniques
        2. **Roast Type**: Considers the roasting style of the coffee
        3. **Origin**: Takes into account where the coffee beans come from
        4. **Price**: Considers the price point of the coffee
        
        The bar chart above shows how much each feature contributes to the recommendations.
        Higher bars indicate more important features in determining coffee similarity.
        """)






###################################################################
# Conclusions Page ################################################
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
