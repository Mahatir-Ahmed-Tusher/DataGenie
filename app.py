import streamlit as st
import pandas as pd
from faker import Faker
import random
from groq import Groq
from io import BytesIO
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import matplotlib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.groq import Groq as LangChainGroq
import torch
import os

# Conditional import for time series models
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Set matplotlib backend for Streamlit compatibility
matplotlib.use('Agg')

# Initialize Faker and apply custom styles
fake = Faker()

def add_custom_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
            background-color: #f4f4f9;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .header-banner {
            text-align: center;
            margin-bottom: 20px;
        }
        .header-banner img {
            max-width: 150px;
            margin-bottom: 10px;
        }
        .header-banner h1 {
            font-size: 36px;
            color: #333;
            margin: 0;
        }
        .header-banner p {
            font-size: 16px;
            color: #666;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            padding: 10px;
            font-size: 14px;
            color: #888;
        }
        footer a {
            color: #4CAF50;
            text-decoration: none;
        }
        footer a:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def add_header():
    st.markdown(
        """
        <div class="header-banner">
            <img src="https://i.postimg.cc/5y20B10S/89c59ca6-c8a8-4210-ba7b-f77a44a8fa3a-removalai-preview.png" alt="DataGenie Logo" style="max-width: 280px;">
            <p>Empowering your data journey with AI-driven insights and synthetic datasets</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### Upload Your Dataset for Preprocessing, Training, and EDA")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset uploaded successfully!")
            st.session_state['uploaded_df'] = df
            st.write("Preview of the uploaded dataset:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
    else:
        st.info("Upload a CSV file to get started.")

def add_footer():
    st.markdown(
        """
        <footer>
            Developed by <a href="https://github.com/Mahatir-Ahmed-Tusher" target="_blank">Mahatir Ahmed Tusher</a>. 
            Inspired by the project "Predicta" by Ahmed Nafiz.
        </footer>
        """,
        unsafe_allow_html=True
    )

def add_sidebar():
    st.sidebar.image(
        "https://i.postimg.cc/5y20B10S/89c59ca6-c8a8-4210-ba7b-f77a44a8fa3a-removalai-preview.png",
        width=150,
        caption="DataGenie"
    )
    st.sidebar.markdown("---")
    st.sidebar.title("About DataGenie")
    st.sidebar.info(
        "DataGenie: AI-powered data science assistant. Generate datasets, analyze data, build ML models. Features: dataset generation, visualization, outlier detection, feature processing, ML model selection, and chat-based exploration."
    )
    st.sidebar.write("**Developed by:** Mahatir Ahmed Tusher")
    st.sidebar.write("**Inspired by:** Predicta by Ahmed Nafiz")
    st.sidebar.markdown("---")
    st.sidebar.write("**Your**")
    st.sidebar.image(
        "https://i.postimg.cc/5y20B10S/89c59ca6-c8a8-4210-ba7b-f77a44a8fa3a-removalai-preview.png",
        width=150
    )

# App configuration
APP_NAME = "DataGenie"

# Initialize Groq client with API key
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    st.error(f"Invalid Groq API key: {str(e)}. Please set GROQ_API_KEY in environment variables.")
    st.stop()

# Utility functions
def extract_row_count(prompt):
    match = re.search(r'(\d+)\s*(rows|records|entries)', prompt, re.IGNORECASE)
    return int(match.group(1)) if match else 100

def generate_dataset_code(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python code generator specializing in creating synthetic datasets using pandas, faker, and random. "
                        "Based on the user's natural language prompt, generate a valid Python function named `create_dataset()` that returns a pandas DataFrame. "
                        "Follow these strict rules:\n"
                        "1. The function must start exactly with `def create_dataset():` and take no arguments.\n"
                        "2. Use only `pd` (pandas), `fake` (Faker), and `random` (random module) within the function.\n"
                        "3. Extract the number of rows from the prompt (e.g., '500 rows' or '1000 records') and use `range(<row_count>)` to generate exactly that many rows. If no row count is specified, default to 100 rows.\n"
                        "4. Generate realistic data for all columns specified in the prompt, respecting any domain-specific details (e.g., age between 18-80, prices in USD, regional names).\n"
                        "5. For target columns (e.g., 'yes/no', 'percentage', 'price', 'category'), use appropriate distributions or logic (e.g., random.choice(['Yes', 'No']), random.uniform(0, 100) for percentages).\n"
                        "6. Ensure data types are correct: integers for counts, floats for percentages/prices, strings for names/emails, etc.\n"
                        "7. The function must end with `return pd.DataFrame(data)` where `data` is a dictionary of column lists.\n"
                        "8. Do not include comments, markdown, explanations, or extra text outside the function definition.\n"
                        "Example for prompt 'Generate 200 rows of customer data with name, age, email, and purchase_amount':\n"
                        "def create_dataset():\n"
                        "    data = {\n"
                        "        'name': [fake.name() for _ in range(200)],\n"
                        "        'age': [random.randint(18, 80) for _ in range(200)],\n"
                        "        'email': [fake.email() for _ in range(200)],\n"
                        "        'purchase_amount': [round(random.uniform(10.0, 500.0), 2) for _ in range(200)]\n"
                        "    }\n"
                        "    return pd.DataFrame(data)\n"
                        "Handle edge cases gracefully, such as missing column details, by using reasonable defaults. "
                        "Ensure the code is syntactically correct and executable. Remember, in case of classification yes means 1 and no means 0."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model="llama-3.3-70b-versatile",
        )
        code = chat_completion.choices[0].message.content.strip()
        if not code.startswith("def create_dataset():"):
            st.error("Generated code does not define create_dataset function correctly.")
            st.code(code, language="python")
            return None
        try:
            ast.parse(code)
            return code
        except SyntaxError as e:
            st.error(f"Invalid syntax in generated code: {str(e)}")
            st.code(code, language="python")
            return None
    except Exception as e:
        st.error(f"Error with Groq API: {str(e)}")
        return None

def execute_code(code):
    safe_globals = {
        "pd": pd,
        "fake": fake,
        "random": random,
        "__builtins__": {
            "range": range, "list": list, "int": int, "str": str, "float": float,
            "round": round, "True": True, "False": False, "zip": zip,
        },
    }
    safe_locals = {}
    try:
        exec(code, safe_globals, safe_locals)
        create_dataset = safe_locals.get("create_dataset")
        if not create_dataset:
            st.error("No create_dataset function defined.")
            return None
        df = create_dataset()
        if not isinstance(df, pd.DataFrame):
            st.error("Generated code did not return a pandas DataFrame.")
            return None
        return df
    except Exception as e:
        st.error(f"Execution error: {str(e)}")
        return None

def to_csv_bytes(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output

# Visualization functions
def visualize_dataset(df):
    st.subheader("Dataset Visualizations")
    if df.empty or not isinstance(df, pd.DataFrame):
        st.warning("No valid data to visualize.")
        return

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    all_cols = numerical_cols + categorical_cols + datetime_cols
    if not all_cols:
        st.warning("No columns available to visualize.")
        return

    viz_type = st.sidebar.selectbox("Select Visualization Type",
                                    ["Histogram", "Box Plot", "Scatter Plot", "Count Plot",
                                     "Correlation Heatmap"] + (["Time Series"] if datetime_cols and numerical_cols else []))
    plt.clf()

    try:
        if viz_type == "Histogram" and numerical_cols:
            col = st.sidebar.selectbox("Select Numerical Column", numerical_cols)
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=col, kde=True, bins='auto', ax=ax)
            st.pyplot(fig)
            download_image(fig, f"histogram_{col}")
            plt.close(fig)

        elif viz_type == "Box Plot" and numerical_cols:
            col = st.sidebar.selectbox("Select Numerical Column", numerical_cols)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, y=col, ax=ax)
            st.pyplot(fig)
            download_image(fig, f"boxplot_{col}")
            plt.close(fig)

        elif viz_type == "Scatter Plot" and len(numerical_cols) >= 2:
            x_col = st.sidebar.selectbox("Select X-axis Column", numerical_cols)
            y_col = st.sidebar.selectbox("Select Y-axis Column", [c for c in numerical_cols if c != x_col])
            fig = px.scatter(df, x=x_col, y=y_col)
            st.plotly_chart(fig)
            img_bytes = io.BytesIO()
            fig.write_image(img_bytes, format='png')
            st.sidebar.download_button("Download Scatter Plot", img_bytes.getvalue(),
                                       file_name=f"scatter_{x_col}_{y_col}.png",
                                       key=f"scatter_{x_col}_{y_col}_{datetime.now().strftime('%H%M%S')}")
        
        elif viz_type == "Count Plot" and categorical_cols:
            col = st.sidebar.selectbox("Select Categorical Column", categorical_cols)
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=col, ax=ax)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            download_image(fig, f"countplot_{col}")
            plt.close(fig)

        elif viz_type == "Correlation Heatmap" and numerical_cols:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt='.2f', ax=ax)
            st.pyplot(fig)
            download_image(fig, "correlation_heatmap")
            plt.close(fig)

        elif viz_type == "Time Series" and datetime_cols and numerical_cols:
            datetime_col = st.sidebar.selectbox("Select Datetime Column", datetime_cols)
            value_col = st.sidebar.selectbox("Select Value Column", numerical_cols)
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
            fig = px.line(df, x=datetime_col, y=value_col)
            st.plotly_chart(fig)
            img_bytes = io.BytesIO()
            fig.write_image(img_bytes, format='png')
            st.sidebar.download_button("Download Time Series", img_bytes.getvalue(),
                                       file_name=f"time_series_{datetime_col}_{value_col}.png",
                                       key=f"timeseries_{datetime_col}_{value_col}_{datetime.now().strftime('%H%M%S')}")
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

def visualize_specific_features(df, features):
    st.subheader("Feature-Specific Visualizations")
    for feature in features:
        if feature not in df.columns:
            st.warning(f"Feature '{feature}' not found.")
            continue
        fig, ax = plt.subplots()
        try:
            if pd.api.types.is_numeric_dtype(df[feature]):
                sns.histplot(data=df, x=feature, kde=True, bins='auto', ax=ax)
            elif pd.api.types.is_categorical_dtype(df[feature]) or pd.api.types.is_string_dtype(df[feature]):
                sns.countplot(data=df, x=feature, ax=ax)
                plt.xticks(rotation=45, ha='right')
            elif pd.api.types.is_datetime64_any_dtype(df[feature]):
                st.warning(f"Use 'Time Series' in main visualization for '{feature}'.")
                plt.close(fig)
                continue
            st.pyplot(fig)
            download_image(fig, f"feature_{feature}")
            plt.close(fig)
        except Exception as e:
            st.error(f"Error visualizing '{feature}': {str(e)}")
            plt.close(fig)

def download_image(fig, key_prefix):
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    st.sidebar.download_button(label="Download Image", data=img_bytes,
                               file_name=f"{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                               mime="image/png",
                               key=f"download_{key_prefix}_{datetime.now().strftime('%H%M%S')}")

# Data processing functions
def dataset_overview(df):
    st.subheader("Dataset Overview")
    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    st.write("Data Types:", df.dtypes)
    st.write(df.head())

def clean_data(df):
    st.subheader("Clean Data")
    cleaned_df = df.dropna().drop_duplicates()
    st.write("Cleaned Dataset:", cleaned_df.head())
    return cleaned_df

def detect_outlier(df):
    st.subheader("Detect Outliers")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        if not outliers.empty:
            st.write(f"Outliers in {col}:", outliers)

def encoder(df):
    st.subheader("Encode Data")
    le = LabelEncoder()
    encoded_df = df.copy()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        encoded_df[col] = le.fit_transform(df[col])
    st.write("Encoded Dataset:", encoded_df.head())
    return encoded_df

def data_transformer(df):
    st.subheader("Data Transformer")
    transformed_df = df.copy()  # Placeholder for future transformations
    st.write("Transformed Dataset:", transformed_df.head())
    return transformed_df

def data_analysis(df):
    st.subheader("Data Analysis")
    st.write(df.describe())

def feature_importance_analyzer(df):
    st.subheader("Feature Importance Analyzer")
    target_column = st.selectbox("Select Target Column", df.columns)
    feature_columns = [col for col in df.columns if col != target_column]
    if not feature_columns:
        st.warning("No features available.")
        return

    X = pd.get_dummies(df[feature_columns], drop_first=True)
    y = df[target_column]
    if y.dtype in ['object', 'category']:
        y = LabelEncoder().fit_transform(y)

    try:
        model = RandomForestClassifier(random_state=42) if y.nunique() <= 10 else RandomForestRegressor(random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
        st.write("Feature Importances:", importance_df)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis", ax=ax)
        st.pyplot(fig)
        download_image(fig, "feature_importance")
        plt.close(fig)
    except Exception as e:
        st.error(f"Error analyzing features: {str(e)}")

def best_parameter_selector(df):
    st.subheader("Best Parameter Selector")
    task_type = st.selectbox("Select Task Type", ["Classification", "Regression"])
    target_column = st.selectbox("Select Target Column", df.columns)
    feature_columns = [col for col in df.columns if col != target_column]
    if not feature_columns:
        st.warning("No features available.")
        return

    X = pd.get_dummies(df[feature_columns], drop_first=True)
    y = df[target_column]
    if task_type == "Classification" and y.dtype in ['object', 'category']:
        y = LabelEncoder().fit_transform(y)

    model_options = {
        "Classification": {
            "Logistic Regression": (LogisticRegression, {"C": [0.01, 0.1, 1], "max_iter": [100, 200]}),
            "Random Forest": (RandomForestClassifier, {"n_estimators": [50, 100], "max_depth": [None, 10]}),
            "SVM": (SVC, {"C": [0.1, 1], "kernel": ["rbf", "linear"]})
        },
        "Regression": {
            "Linear Regression": (LinearRegression, {}),
            "Random Forest": (RandomForestRegressor, {"n_estimators": [50, 100], "max_depth": [None, 10]}),
            "SVR": (SVR, {"C": [0.1, 1], "epsilon": [0.1, 0.2]})
        }
    }
    model_name = st.selectbox("Select Model", list(model_options[task_type].keys()))
    model_class, param_grid = model_options[task_type][model_name]
    model = model_class(random_state=42) if "random_state" in model_class.__init__.__code__.co_varnames else model_class()

    for param, values in param_grid.items():
        new_values = st.text_input(f"Values for {param} (comma-separated)", ",".join(map(str, values)) if values else "")
        if new_values:
            param_grid[param] = [float(x) if '.' in x else int(x) for x in new_values.split(',')]

    scoring = st.selectbox("Select Scoring Metric", ["accuracy", "f1"] if task_type == "Classification" else ["r2", "neg_mean_squared_error"])
    try:
        if param_grid:
            with st.spinner("Performing GridSearchCV..."):
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
                grid_search.fit(X, y)
            st.write("Best Parameters:", grid_search.best_params_)
            st.write("Best Score:", grid_search.best_score_)
        else:
            model.fit(X, y)
            st.write("Model trained with default parameters. Score:", model.score(X, y))
    except Exception as e:
        st.error(f"Parameter selection error: {str(e)}")

def select_ml_models(df):
    st.subheader("Select ML Models")
    analysis_type = st.selectbox("Select Analysis Type", ["Classification", "Regression", "Clustering", "Time Series"])
    
    if analysis_type in ["Classification", "Regression"]:
        target_col = st.selectbox("Select Target Variable", df.columns)
        feature_cols = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_col])
        if not feature_cols:
            st.warning("Select at least one feature.")
            return

        X = pd.get_dummies(df[feature_cols])
        y = df[target_col]

        if analysis_type == "Classification":
            if pd.api.types.is_float_dtype(y) or (pd.api.types.is_numeric_dtype(y) and y.nunique() > len(y) // 10):
                st.error(
                    f"Target column '{target_col}' appears to be continuous (float or many unique values: {y.nunique()}). "
                    "Classification requires discrete labels (e.g., 'Yes/No', integers with few unique values). "
                    "Please select a categorical target, bin this column, or choose 'Regression' for continuous targets."
                )
                return
            if y.dtype in ['object', 'category'] or pd.api.types.is_string_dtype(y):
                y = LabelEncoder().fit_transform(y)
        elif analysis_type == "Regression":
            if not pd.api.types.is_numeric_dtype(y):
                st.error(
                    f"Target column '{target_col}' is not numeric (type: {y.dtype}). "
                    "Regression requires a numeric target (e.g., float or integer). "
                    "Please select a numeric target or preprocess the data."
                )
                return

        model_options = {
            "Classification": {
                "Logistic Regression": LogisticRegression(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC(random_state=42),
                "KNN": KNeighborsClassifier()
            },
            "Regression": {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "SVR": SVR(),
                "Decision Tree": DecisionTreeRegressor(random_state=42)
            }
        }[analysis_type]

        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = model_options[selected_model]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = {
                        "Classification": {
                            "Accuracy": accuracy_score(y_test, y_pred),
                            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        },
                        "Regression": {
                            "MAE": mean_absolute_error(y_test, y_pred),
                            "MSE": mean_squared_error(y_test, y_pred),
                            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                            "R²": r2_score(y_test, y_pred)
                        }
                    }[analysis_type]
                    st.write("Model Performance:", pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))
                except Exception as e:
                    st.error(f"Training error: {str(e)}")

    elif analysis_type == "Clustering":
        feature_cols = st.multiselect("Select Features for Clustering", df.columns)
        if not feature_cols:
            st.warning("Select at least one feature.")
            return

        X = pd.get_dummies(df[feature_cols])
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        clustering_models = {
            "K-Means": KMeans(n_clusters=n_clusters, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
            "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters)
        }
        selected_model = st.selectbox("Select Clustering Algorithm", list(clustering_models.keys()))
        if st.button("Perform Clustering"):
            with st.spinner("Performing clustering..."):
                X_scaled = StandardScaler().fit_transform(X)
                model = clustering_models[selected_model]
                clusters = model.fit_predict(X_scaled)
                df_clusters = df.copy()
                df_clusters['Cluster'] = clusters
                st.write("Clustered Data Sample:", df_clusters.head())
                if selected_model != "DBSCAN":
                    metrics = {
                        "Silhouette": silhouette_score(X_scaled, clusters),
                        "Davies-Bouldin": davies_bouldin_score(X_scaled, clusters),
                        "Calinski-Harabasz": calinski_harabasz_score(X_scaled, clusters)
                    }
                    st.write("Clustering Metrics:", pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

    elif analysis_type == "Time Series":
        if not HAS_STATSMODELS:
            st.error("Install statsmodels: `pip install statsmodels`")
            return
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if not datetime_cols.empty:
            date_col = st.selectbox("Select Date Column", datetime_cols)
            value_col = st.selectbox("Select Value Column", df.select_dtypes(include=['float64', 'int64']).columns)
            forecast_models = {"Exponential Smoothing": ExponentialSmoothing, "ARIMA": ARIMA}
            selected_model = st.selectbox("Select Forecasting Model", list(forecast_models.keys()))
            if st.button("Analyze Time Series"):
                with st.spinner("Analyzing time series..."):
                    ts_df = df.sort_values(date_col)
                    train_size = int(len(ts_df) * 0.8)
                    train, test = ts_df[:train_size], ts_df[train_size:]
                    if selected_model == "Exponential Smoothing":
                        model = ExponentialSmoothing(train[value_col], trend='add', seasonal='add', seasonal_periods=12).fit()
                    else:
                        model = ARIMA(train[value_col], order=(1, 1, 1)).fit()
                    forecast = model.forecast(steps=len(test))
                    metrics = {
                        "MAE": mean_absolute_error(test[value_col], forecast),
                        "MSE": mean_squared_error(test[value_col], forecast),
                        "RMSE": np.sqrt(mean_squared_error(test[value_col], forecast)),
                        "MAPE": np.mean(np.abs((test[value_col] - forecast) / test[value_col])) * 100
                    }
                    st.write("Forecasting Metrics:", pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

def clear_modified_dataset():
    st.subheader("Clear Modified Dataset")
    st.session_state.pop('uploaded_df', None)
    st.write("Dataset cleared.")

def chat_with_dataset(df):
    st.subheader("Chat with Your Dataset")
    st.write("Ask questions about your dataset. For example, 'What is the average value of column X?' or 'Show me the top 5 rows.'")

    user_query = st.text_area("Enter your query:", height=100)
    if st.button("Ask"):
        if not user_query.strip():
            st.warning("Please enter a query.")
            return

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert data analyst. Answer the user's questions about the provided pandas DataFrame. "
                            "Use Python pandas to analyze the data and provide concise answers. "
                            "If the user asks for code, generate Python code snippets using pandas to perform the requested operation. "
                            "Do not include explanations unless explicitly requested."
                        ),
                    },
                    {"role": "user", "content": f"The dataset is:\n{df.head(5).to_string()}\n\n{user_query}"},
                ],
                model="llama-3.3-70b-versatile",
            )
            response = chat_completion.choices[0].message.content.strip()
            st.write("Response:")
            st.code(response, language="python" if "def " in response or "import " in response else None)

            st.write("You can execute the generated code below:")
            if st.button("Execute Generated Code"):
                try:
                    safe_globals = {"pd": pd, "plt": plt, "sns": sns, "df": df, "io": io, "np": np}
                    safe_locals = {}
                    exec(response, safe_globals, safe_locals)
                    
                    # Check for matplotlib or seaborn plots
                    if "plt." in response or "sns." in response:
                        st.pyplot(plt.gcf())
                        plt.clf()
                    
                    # Check for DataFrame outputs
                    elif "pd.DataFrame" in response or "df" in response:
                        output_df = safe_locals.get("df", None)
                        if isinstance(output_df, pd.DataFrame):
                            st.write("Generated DataFrame:")
                            st.dataframe(output_df)
                        else:
                            st.write("Code executed successfully. Check the output above if applicable.")
                    else:
                        st.write("Code executed successfully. Check the output above if applicable.")
                except Exception as e:
                    st.error(f"Error executing code: {str(e)}")
        except Exception as e:
            st.error(f"Error with Groq API: {str(e)}")

def process_paper_with_rag(uploaded_paper):
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(uploaded_paper)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings (no HF token required)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        # Create vector store
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

        # Initialize Groq LLM for LangChain
        llm = LangChainGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5,
            max_tokens=512
        )

        # Create conversation chain
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        return text, chunks, conversation_chain

    except Exception as e:
        st.error(f"Error processing paper: {str(e)}")
        return None, None, None

def analyze_research_paper():
    st.header("Analyze Research Paper")
    st.write("Upload a research paper (PDF format) to analyze and generate possible code implementations based on the paper's content.")
    
    # Add installation instructions
with st.expander("Setup Instructions"):
    st.write("""
    Before using this feature, please install the required packages:
    ```bash
    pip install PyPDF2 langchain langchain-community faiss-cpu sentence-transformers torch
    """)

uploaded_paper = st.file_uploader("Upload Research Paper (PDF)", type="pdf")
if uploaded_paper:
    try:
        text, chunks, conversation_chain = process_paper_with_rag(uploaded_paper)

        if text and chunks and conversation_chain:
            st.success("Research paper processed successfully!")

            # Show paper chunks
            with st.expander("View Paper Chunks"):
                for i, chunk in enumerate(chunks):
                    st.write(f"Chunk {i+1}:")
                    st.text(chunk)

            if st.button("Generate The Possible Code of the Paper"):
                with st.spinner("Analyzing paper and generating code..."):
                    # Use conversation chain to generate code
                    response = conversation_chain({"question": "Based on this research paper, generate a detailed Python implementation of the main algorithms and methods described. Include all necessary imports and ensure the code is well-structured."})

                    generated_code = response['answer']

                    st.subheader("Generated Code")
                    st.code(generated_code, language="python")

                    # Allow users to download the generated code
                    txt_bytes = BytesIO()
                    txt_bytes.write(generated_code.encode())
                    txt_bytes.seek(0)
                    st.download_button(
                        label="Download Code as TXT",
                        data=txt_bytes,
                        file_name="generated_code.txt",
                        mime="text/plain"
                    )

                    # Store conversation in session state
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append(("user", "Generate code implementation"))
                    st.session_state.chat_history.append(("assistant", generated_code))

            # Add follow-up questions section
            st.subheader("Ask Questions About the Implementation")
            user_question = st.text_input("Enter your question about the paper or implementation:")
            if user_question and st.button("Ask"):
                with st.spinner("Generating response..."):
                    response = conversation_chain({"question": user_question})
                    st.write("Response:", response['answer'])
                    st.session_state.chat_history.append(("user", user_question))
                    st.session_state.chat_history.append(("assistant", response['answer']))

    except Exception as e:
        st.error(f"Error processing the research paper: {str(e)}")
        st.write("Please make sure you have installed all required packages:")
        st.code("pip install PyPDF2 langchain langchain-community faiss-cpu sentence-transformers torch")
else:
    st.info("Upload a research paper to get started.")

# Main app layout
add_custom_styles()
st.title("")
add_header()

tab1, tab2, tab3, tab4 = st.tabs(["Dataset Generator", "Example Prompts", "Chat with Dataset", "Analyze Research Paper"])

with tab1:
    st.header("Generate Synthetic Datasets")
    st.write("Enter a prompt to generate a synthetic dataset. Be as descriptive as possible (e.g., 'Generate 500 rows for heart risk prediction with age, common symptoms like chest pain and shortness of breath, and a risk level (yes/no)'). For more examples, check the 'Example Prompts' tab.")
    prompt = st.text_area("Your prompt:", height=100)

    if "generated_code" not in st.session_state:
        st.session_state.generated_code = None
        st.session_state.expected_rows = None

    if st.button("Generate Code"):
        if prompt:
            code = generate_dataset_code(prompt)
            if code:
                st.session_state.generated_code = code
                st.session_state.expected_rows = extract_row_count(prompt)
                st.subheader("Generated Python Code")
                st.code(code, language="python")
                st.info("Review the code and click 'Get the Dataset'.")
            else:
                st.error("Generated code does not define create_dataset function correctly.")
        else:
            st.warning("Enter a prompt.")

    if st.session_state.generated_code and st.button("Get the Dataset"):
        df = execute_code(st.session_state.generated_code)
        if df is not None:
            if len(df) != st.session_state.expected_rows:
                st.warning(f"Dataset has {len(df)} rows; requested {st.session_state.expected_rows}.")
            st.subheader("Generated Dataset")
            st.write(f"Rows: {len(df)}, Columns: {', '.join(df.columns)}")
            st.dataframe(df.head())
            csv_bytes = to_csv_bytes(df)
            st.download_button(label="Download CSV", data=csv_bytes, file_name="datagenie_dataset.csv", mime="text/csv")

with tab2:
    st.header("Example Prompts")
    st.write("Explore example prompts to generate synthetic datasets for various domains.")
    st.subheader("💼 Finance & Business")
    st.write("Generate 1000 customer records for a bank with age, income, loan amount, credit score, and defaulted (Yes/No).")
    st.write("Create 500 rows of sales data with product category, region, sales amount, profit margin, and sales channel (Online/Offline).")
    st.write("Generate 200 rows of stock market data with date, opening price, closing price, highest price, lowest price, and trading volume.")

    st.subheader("🧑‍🎓 Education")
    st.write("Create 700 student records with study hours, attendance, and final grade (A, B, C, D, F).")
    st.write("Generate 300 rows of teacher performance data with years of experience, subject taught, average student score, and teacher rating (1-5).")
    st.write("Generate 1000 rows of university admission data with applicant age, GPA, SAT score, extracurricular activities, and admission status (Accepted/Rejected).")

    st.subheader("🌍 Environment")
    st.write("Generate 365 days of air quality data with PM2.5, PM10, CO2, and air quality (Good, Moderate, Hazardous).")
    st.write("Create 500 rows of weather data with date, temperature, humidity, wind speed, and precipitation level.")
    st.write("Generate 1000 rows of energy consumption data with household size, monthly usage (kWh), energy source (Solar, Wind, Grid), and cost.")

    st.subheader("🏥 Healthcare")
    st.write("Generate 1000 patient records with age, gender, blood pressure, cholesterol level, and diagnosis (Healthy, At Risk, Critical).")
    st.write("Create 500 rows of hospital data with department, number of patients, average treatment cost, and satisfaction rating (1-5).")
    st.write("Generate 300 rows of clinical trial data with participant ID, age, treatment type, side effects (Yes/No), and outcome (Improved/Unchanged/Worsened).")

    st.subheader("🚗 Transportation")
    st.write("Generate 1000 rows of vehicle data with make, model, year, fuel efficiency (mpg), and price.")
    st.write("Create 500 rows of traffic data with date, time, location, number of vehicles, and average speed.")
    st.write("Generate 300 rows of ride-sharing data with driver ID, trip distance, trip duration, fare amount, and rating (1-5).")

    st.subheader("🛒 Retail & E-commerce")
    st.write("Generate 1000 rows of customer purchase data with customer ID, product category, purchase amount, and payment method (Credit Card, PayPal, Cash).")
    st.write("Create 500 rows of inventory data with product ID, category, stock level, reorder point, and supplier.")
    st.write("Generate 300 rows of website analytics data with date, page views, unique visitors, bounce rate, and conversion rate.")

    st.subheader("🏗️ Construction & Real Estate")
    st.write("Generate 500 rows of real estate data with property type, location, size (sq ft), price, and status (Available/Sold).")
    st.write("Create 300 rows of construction project data with project ID, start date, end date, budget, and completion status (On Track/Delayed).")
    st.write("Generate 200 rows of rental data with property type, monthly rent, tenant age, and lease duration (months).")

    st.subheader("🎮 Gaming & Entertainment")
    st.write("Generate 1000 rows of gaming data with player ID, game title, hours played, in-game purchases, and player rank.")
    st.write("Create 500 rows of movie data with title, genre, release year, box office revenue, and IMDb rating.")
    st.write("Generate 300 rows of music streaming data with user ID, song title, artist, play count, and duration (minutes).")

with tab3:
    st.header("Chat with Dataset")
    uploaded_file = st.file_uploader("Upload CSV for Chatting", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            chat_with_dataset(df)
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
    else:
        st.info("Upload a CSV file to start chatting.")

with tab4:
    analyze_research_paper()

add_footer()

# Sidebar for data processing and visualization
add_sidebar()
feature_options = st.sidebar.radio("Select Option", ["Dataset Overview", "Clean Data", "Detect Outlier", "Encoder",
                                                    "Data Transformer", "Data Analysis", "Feature Importance Analyzer",
                                                    "Best Parameter Selector", "Select ML Models", "Clear Modified Dataset",
                                                    "Visualizations"])

if 'uploaded_df' in st.session_state:
    df = st.session_state['uploaded_df']
    try:
        if feature_options == "Dataset Overview":
            dataset_overview(df)
        elif feature_options == "Clean Data":
            st.session_state['uploaded_df'] = clean_data(df)
        elif feature_options == "Detect Outlier":
            detect_outlier(df)
        elif feature_options == "Encoder":
            st.session_state['uploaded_df'] = encoder(df)
        elif feature_options == "Data Transformer":
            st.session_state['uploaded_df'] = data_transformer(df)
        elif feature_options == "Data Analysis":
            data_analysis(df)
        elif feature_options == "Feature Importance Analyzer":
            feature_importance_analyzer(df)
        elif feature_options == "Best Parameter Selector":
            best_parameter_selector(df)
        elif feature_options == "Select ML Models":
            select_ml_models(df)
        elif feature_options == "Clear Modified Dataset":
            clear_modified_dataset()
        elif feature_options == "Visualizations":
            visualize_dataset(df)
            features = st.sidebar.multiselect("Select features for specific visualizations", df.columns.tolist())
            if features:
                visualize_specific_features(df, features)

        if 'uploaded_df' in st.session_state:
            df = st.session_state['uploaded_df']
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
else:
    st.sidebar.info("Upload a CSV to proceed.")
