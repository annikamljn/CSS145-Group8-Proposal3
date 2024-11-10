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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
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

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

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

# Pages

###################################################################
# About Page ######################################################
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here

###################################################################
# Dataset Page ####################################################
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("Coffee Reviews Dataset by `@schmoyote`")
    st.write("")

    # Your content for your DATASET page goes here

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
    missing_values = round((df.isnull().sum()/df.shape[0])*100, 2)
    st.write(missing_values)

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

    
    st.write('Lower Bound:', price_lower_bound)
    st.write('Upper Bound:', price_upper_bound)

    # Filter the dataset to remove outliers
    df = df[(df['100g_USD'] >= price_lower_bound) & (df['100g_USD'] <= price_upper_bound)]

    # Display price statistics
    st.markdown("**Price Statistics after Outlier Removal:**")
    st.write(df['100g_USD'].describe())

    # Rating Outliers
    st.subheader("Rating Outliers")
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

    st.write('Lower Bound:', rating_lower_bound)
    st.write('Upper Bound:', rating_upper_bound)

    # Filter the dataset to remove outliers
    df = df[(df['rating'] >= rating_lower_bound) & (df['rating'] <= rating_upper_bound)]

    # Display rating statistics
    st.markdown("**Rating Statistics after Outlier Removal:**")
    st.write(df['rating'].describe())

    ##### Text pre-processing
    st.subheader("Coffee review text pre-processing", divider=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    st.markdown("`df[['desc_1', 'desc_2', 'desc_3']].head()`")
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

        with col[0]:
            with st.expander('Legend', expanded=True):
                st.write('''
                    - Data: [Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset).
                    - :orange[**Pie Chart**]: TBA.
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
            

    else:
        st.error("DataFrame not found. Please go to the Data Cleaning page first. (It is required that you scroll to the bottom)")

###################################################################
# Machine Learning Page ###########################################
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

###################################################################
# Prediction Page #################################################
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

###################################################################
# Conclusions Page ################################################
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here