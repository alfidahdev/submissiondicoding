import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your datasets
@st.cache_data
def load_data():
    day_data = pd.read_csv('data/processed_day.csv')
    hour_data = pd.read_csv('data/processed_hour.csv')
    return day_data, hour_data

day, hour = load_data()

# Sidebar
st.sidebar.image('images/dicoding.jpg', caption="This project aims to fulfill the final project requirement from the Dicoding course 'Belajar Analisis Data dengan Python'")

st.sidebar.header("Filters")

# Season Filter
season_filter = st.sidebar.selectbox("Select Season", ["", "Spring", "Summer", "Fall", "Winter"], index=0)
season_mapping = {"":None, "Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

if season_filter:
    selected_season_code = season_mapping[season_filter]
else:
    selected_season_code = None

# Datetime Filter
start_date = pd.to_datetime("2011-01-01")
end_date = pd.to_datetime("2012-12-31")
datetime_filter = st.sidebar.date_input("Select Date Range", [start_date, end_date], min_value=start_date, max_value=end_date)

if len(datetime_filter) == 2:
    start_date = pd.to_datetime(datetime_filter[0])
    end_date = pd.to_datetime(datetime_filter[1])
else:
    start_date = None
    end_date = None

# Apply Filters
if selected_season_code is not None and start_date is not None and end_date is not None:
    day = day[(day['season'] == selected_season_code) & (pd.to_datetime(day['dteday']) >= start_date) & (pd.to_datetime(day['dteday']) <= end_date)]
    hour = hour[(hour['season'] == selected_season_code) & (pd.to_datetime(hour['dteday']) >= start_date) & (pd.to_datetime(hour['dteday']) <= end_date)]
elif selected_season_code is not None:
    day = day[(day['season'] == selected_season_code)]
    hour = hour[(hour['season'] == selected_season_code)]
elif start_date is not None and end_date is not None:
    day = day[(pd.to_datetime(day['dteday']) >= start_date) & (pd.to_datetime(day['dteday']) <= end_date)]
    hour = hour[(pd.to_datetime(hour['dteday']) >= start_date) & (pd.to_datetime(hour['dteday']) <= end_date)]


st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Exploratory Data Analysis", "Explanatory Data Analysis"])

# Introduction
if page == "Introduction":
    st.image('images/bangkit.png', caption="Bangkit Academy 2024")

    col1, col2 = st.columns([1, 3])
    
    st.write("### Daily Data")
    st.write(day.head())

    st.write("### Hourly Data")
    st.write(hour.head())

    st.write("""
    ### Dataset Information

    - `instant`: Record index
    - `dteday`: Date
    - `season`: Season (1: spring, 2: summer, 3: fall, 4: winter)
    - `yr`: Year (0: 2011, 1: 2012)
    - `mnth`: Month (1 to 12)
    - `hr`: Hour (0 to 23)
    - `holiday`: Weather day is holiday or not
    - `weekday`: Day of the week
    - `workingday`: If day is neither weekend nor holiday is 1, otherwise is 0
    - `weathersit`: 
        - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
        - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
        - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
        - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
    - `temp`: Normalized temperature in Celsius (values divided by 41, max)
    - `atemp`: Normalized feeling temperature in Celsius (values divided by 50, max)
    - `hum`: Normalized humidity (values divided by 100, max)
    - `windspeed`: Normalized wind speed (values divided by 67, max)
    - `casual`: Count of casual users
    - `registered`: Count of registered users
    - `cnt`: Count of total rental bikes including both casual and registered
    """)

elif page == "Exploratory Data Analysis":
    st.write("""
    ## Exploratory Data Analysis
    """)
    st.write("Select tabs")
    tab1, tab2, tab3, tab4 = st.tabs(["Day and Hour Descriptive Statistics", "Numeric Columns Distribution", "Numeric Columns Correlation", "Clustering Monthly Bike Usage"])
    with tab1:
        st.write("### Day Descriptive Statistics")
        st.write(day.describe())

        st.write("### Hour Descriptive Statistics")
        st.write(hour.describe())

    with tab2:
        st.write("### Day Numeric Columns Distribution (Continuous)")
        numeric_columns = day.drop(columns=['instant', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']).select_dtypes(include=['number']).columns
        n_cols = 4
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        fig.suptitle('Distribution of Day Numeric Columns (Continuous)', fontsize=16)

        for i, column in enumerate(numeric_columns):
            row, col = divmod(i, n_cols)
            sns.histplot(day[column], kde=True, ax=axes[row, col])
            axes[row, col].set_title(f'Distribution of {column}')
            axes[row, col].set_xlabel(column)
            axes[row, col].set_ylabel('Frequency')

        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Hour Numeric Columns Distribution (Continuous)")
        numeric_columns = hour.drop(columns=['instant', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']).select_dtypes(include=['number']).columns
        n_cols = 4
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        fig.suptitle('Distribution of Hour Numeric Columns (Continuous)', fontsize=16)

        for i, column in enumerate(numeric_columns):
            row, col = divmod(i, n_cols)
            sns.histplot(hour[column], kde=True, ax=axes[row, col])
            axes[row, col].set_title(f'Distribution of {column}')
            axes[row, col].set_xlabel(column)
            axes[row, col].set_ylabel('Frequency')

        plt.tight_layout()
        st.pyplot(fig)

        
        st.write("""
        The data exhibits normal distributions and some skewness. An interesting pattern emerges in the hourly visualizations, where spikes for casual, registered, and cnt at maximum values are observed. This may be influenced by the imputation of outlier values previously conducted.
                 """)
        
        
        st.write("""
        For the discrete numeric columns, we will focus on examining the distribution of `holiday`, `workingday`, and `weathersit` solely from the 'day' dataframe. The rationale for not analyzing variables such as `season`, `year`, `month`, and `weekday` is that, given the daily nature of the data, the results would remain consistent.
        Additionally, there is no need to assess the 'hour' dataframe, as it would merely increase the count of observations without altering the distribution of variables like `holiday`, `workingday`, and `weathersit`, which remain constant throughout the hours.
                 """)

        st.write("### Numeric Columns Distribution (Discrete)")
        numeric_columns = ['holiday', 'workingday', 'weathersit']
        n_cols = 3
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        fig.suptitle('Count Distribution of Day Numeric Columns (Discrete)', fontsize=16)

        axes = axes.flatten()

        for i, column in enumerate(numeric_columns):
            sns.countplot(x=column, data=day, ax=axes[i])
            axes[i].set_title(f'Count of {column}')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Count')

        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.write("### Correlation Matrix Heatmap")
        corr_day = day[day.drop(columns=['instant']).select_dtypes(include=['number']).columns].corr()
        corr_hour = hour[hour.drop(columns=['instant']).select_dtypes(include=['number']).columns].corr()

        mask_day = np.triu(np.ones_like(corr_day, dtype=bool))
        mask_hour = np.triu(np.ones_like(corr_hour, dtype=bool))
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))

        sns.heatmap(corr_day, mask=mask_day, annot=True, fmt=".2f", cmap='coolwarm',
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[0])
        ax[0].set_title('Day Correlation Matrix Heatmap')

        sns.heatmap(corr_hour, mask=mask_hour, annot=True, fmt=".2f", cmap='coolwarm',
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[1])
        ax[1].set_title('Hour Correlation Matrix Heatmap')

        plt.tight_layout()
        st.pyplot(fig)

        st.write("""
        Although we should use the appropriate correlation method for categorical variables, the materials did not cover this aspect. Therefore, we will assume that all data can be analyzed using the default correlation method corr().
        We observe that temperature, season, year, and hour play roles in affecting bike usage, as indicated by the correlation with the 'cnt' column.
                 """)

    with tab4:
        st.write("### Monthly Bike Usage Distribution (Category)")

        monthly_rentals = day.groupby(by='mnth').agg({
            'cnt': 'sum',
        }).reset_index()

        def classify_usage(cnt):
            if cnt > 300000:
                return 'High'
            elif cnt > 200000:
                return 'Medium'
            else:
                return 'Low'

        monthly_rentals['category'] = monthly_rentals['cnt'].apply(classify_usage)

        plt.figure(figsize=(10, 6))

        sns.barplot(x='mnth', y='cnt', hue='category', data=monthly_rentals, palette='deep')
        plt.title('Explore Monthly Bike Usage Distribution')
        plt.xlabel('Month')
        plt.ylabel('Total Bike Rentals')
        plt.legend(title='Category')

        st.pyplot(plt)

        st.write("""
        The data has become more insightful. We can observe that mid-year experiences the highest bike usage, while the period towards the end and before the mid-year shows medium usage, and the early part of the year has the lowest.
                 """)

elif page == "Explanatory Data Analysis":
    st.write("""
    ## Explanatory Data Analysis
    """)
    st.write("Select tabs")
    tab1, tab2, tab3 = st.tabs(["Question 1", "Question 2", "Conclusion"])
    with tab1:
        st.write(""" 
        ## How does daily bike usage change over time, and which season or month experiences the highest usage? Is there a correlation between bike usage and the number of holidays in that season or month?
        """)
        st.write("### Daily Bike Usage Over Time")
        plt.figure(figsize=(15, 7))
        plt.plot(day['dteday'], day['cnt'], label='Total', color='blue')
        plt.plot(day['dteday'], day['registered'], label='Registered', color='green')
        plt.plot(day['dteday'], day['casual'], label='Casual', color='red')

        plt.title('Daily Bike Usage Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Bike Rentals')
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        st.write("""
        The daily line plot reveals that **bike usage peaked in the mid to late year**. This correlates with the earlier exploratory analysis, where mid-year exhibited the highest usage. There is no distinct difference between registered and casual users; they exhibit similar patterns, albeit with casual users showing lower counts.
        """)

        st.write("### Explore Monthly Bike Usage Distribution")

        col1, col2 = st.columns([1, 3])

        monthly_rentals = day.groupby(by='mnth').agg({
            'cnt': 'sum',
            'holiday': 'sum'
        }).reset_index()

        with col1:
            st.write(monthly_rentals)
        
        with col2:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='mnth', y='cnt', data=monthly_rentals, palette='pastel', hue='mnth', legend=False)
            plt.title('Explore Monthly Bike Usage Distribution')
            plt.xlabel('Month')
            plt.ylabel('Total Bike Rentals')
            st.pyplot(plt)

        st.write("### Explore Seasonly Bike Usage Distribution")

        col1, col2 = st.columns([1, 3])

        seasonly_rentals = day.groupby(by='season').agg({
            'cnt': 'sum',
            'holiday': 'sum'
        }).reset_index()

        with col1:
            st.write(seasonly_rentals)

        with col2:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='season', y='cnt', data=seasonly_rentals, palette='pastel', hue='season', legend=False)
            plt.title('Explore Seasonly Bike Usage Distribution')
            plt.xlabel('Season')
            plt.ylabel('Total Bike Rentals')
            st.pyplot(plt)

        st.write("""
        The visualization illustrates that bike usage peaked in **August** and during the **fall** season when the number of holidays is lower. This suggests that **number holidays do not significantly affect bike usage**, a conclusion supported by previous correlation analysis, which indicated almost no correlation between holidays and bike count
        """)
        
    with tab2:
        st.write(""" 
        ## What are the mean values of temperature (temp), feels-like temperature (atemp), humidity (hum), and wind speed in each season or month? How do these factors correlate with the number of daily bike rentals?
        """)
        st.write("### Scatterplot of Total Bike Count vs Weather Metrics by Month")
        monthly = day.groupby(by='mnth').agg({
            'temp': 'median',
            'atemp': 'median',
            'hum': 'median',
            'windspeed': 'median',
            'cnt': 'sum'
        }).reset_index()

        columns_to_plot = monthly.drop(columns='mnth').columns

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, column in enumerate(columns_to_plot):
            if column != 'cnt':
                sns.scatterplot(x=column, y='cnt', hue='mnth', data=monthly, alpha=0.5, s=500, palette='deep', ax=axes[i])
                axes[i].set_title(f'Scatterplot of Total Bike Count vs {column} with Month as Hue')
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Total Bike Usage')
                axes[i].grid(True)

        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Scatterplot of Total Bike Count vs Weather Metrics by Season")

        seasonly = day.groupby(by='season').agg({
            'temp': 'median',
            'atemp': 'median',
            'hum': 'median',
            'windspeed': 'median',
            'cnt': 'sum'
        }).reset_index()

        columns_to_plot = seasonly.drop(columns='season').columns

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, column in enumerate(columns_to_plot):
            if column != 'cnt':
                sns.scatterplot(x=column, y='cnt', hue='season', data=seasonly, alpha=0.5, s=500, palette='deep', ax=axes[i])
                axes[i].set_title(f'Scatterplot of Total Bike Count vs {column} with Season as Hue')
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Total Bike Usage')
                axes[i].grid(True)

        plt.tight_layout()
        st.pyplot(fig)

        st.write(
        """
        The plot shows that there is a **moderate correlation between temp, atemp, and cnt**, and a **weak negative correlation between windspeed and cnt**. Humidity, on the other hand, almost doesn't correlate with cnt. This is supported by the exploratory data analysis with the correlation values.
        """
        )

    with tab3:
        st.write("""
        Question 1:
        - The analysis reveals that bike usage reached its peak during the mid to late year, notably in August and the fall season. Surprisingly, holidays did not significantly impact bike usage, as evidenced by the low correlation value, suggesting a lack of correlation between the number of holidays and the total bike count.
                 
        Question 2:
        - Examining temperature patterns, both actual and perceived (as reflected in "atemp"), we observe a similar trend peaking during the mid to late year. Humidity levels also peaked during this period, while windspeed was highest in the early year. The analysis further indicates that higher temperatures coincide with increased bike usage, whereas lower windspeeds correlate with higher bike usage. Currently, there appears to be a moderate level of correlation, though further investigation is warranted to ascertain whether these factors exhibit multicollinearity.
        """)

