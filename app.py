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

# For NLP Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
import nltk
import os
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Ensure spaCy model
try:
    spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")

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
        st.info("Upload a CSV file to get started. And Go to the Sidebar to start working on your dataset")

def add_footer():
    st.markdown(
        """
        <footer>
            Developed by <a href="https://github.com/Mahatir-Ahmed-Tusher" target="_blank">Mahatir Ahmed Tusher</a>. 
            Inspired by the project "Predicta" by <a href="https://github.com/ahammadnafiz" target="_blank"> Ahammad Nafiz </a>.
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
    st.sidebar.write("**Inspired by:** Predicta by Ahammad Nafiz")
    st.sidebar.markdown("---")
    st.sidebar.write("**Your**")
    st.sidebar.image(
        "https://i.postimg.cc/5y20B10S/89c59ca6-c8a8-4210-ba7b-f77a44a8fa3a-removalai-preview.png",
        width=150
    )

# App configuration
APP_NAME = "DataGenie"

# Initialize Groq client with API key
GROQ_API_KEY = "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"
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Invalid Groq API key: {str(e)}. Please update GROQ_API_KEY.")
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
    st.markdown("#### Basic Information")
    st.write(f"**Rows**: {len(df):,} | **Columns**: {len(df.columns):,}")
    st.write(f"**Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    st.markdown("#### Data Types and Missing Values")
    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes,
        "Non-Null Count": df.count(),
        "Missing Values": df.isna().sum(),
        "Missing %": (df.isna().sum() / len(df) * 100).round(2)
    }).reset_index(drop=True)
    st.dataframe(dtypes_df.style.highlight_null(color='lightcoral'))

    st.markdown("#### Numerical Columns Summary")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if numerical_cols.size > 0:
        numerical_summary = df[numerical_cols].describe().T.round(2)
        numerical_summary['Skewness'] = df[numerical_cols].skew().round(2)
        numerical_summary['Kurtosis'] = df[numerical_cols].kurt().round(2)
        st.dataframe(numerical_summary)

    st.markdown("#### Categorical Columns Summary")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if categorical_cols.size > 0:
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(5)
            st.write(f"**{col}** (Top 5 values):")
            st.dataframe(pd.DataFrame({
                "Value": value_counts.index,
                "Count": value_counts.values,
                "% of Total": (value_counts.values / len(df) * 100).round(2)
            }))

    st.markdown("#### Duplicate Rows")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        st.warning(f"Found {duplicate_count} duplicate rows ({duplicate_count / len(df) * 100:.2f}% of total).")
    else:
        st.success("No duplicate rows detected.")

    st.markdown("#### Sample Data (First 5 Rows)")
    st.dataframe(df.head())

def clean_data(df):
    st.subheader("Clean Data")
    st.markdown("#### Missing Values")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        "Missing Values": missing_values,
        "Missing Percentage (%)": missing_percentage
    }).sort_values(by="Missing Values", ascending=False)
    st.dataframe(missing_summary)

    st.markdown("#### Duplicate Rows")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        st.warning(f"Found {duplicate_count} duplicate rows. They will be removed.")
    else:
        st.success("No duplicate rows detected.")

    cleaned_df = df.dropna().drop_duplicates()
    st.write(f"Cleaned Dataset: {len(cleaned_df)} rows remaining after cleaning.")
    st.dataframe(cleaned_df.head())
    return cleaned_df

def detect_outlier(df):
    st.subheader("Detect Outliers")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numerical_cols.any():
        st.warning("No numerical columns available for outlier detection.")
        return

    st.markdown("#### Outlier Detection Summary")
    outlier_summary = []
    for col in numerical_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = round((outlier_count / len(df) * 100), 2)
        outlier_summary.append({
            "Column": col,
            "Outliers": outlier_count,
            "Outlier Percentage (%)": outlier_percentage
        })

    outlier_df = pd.DataFrame(outlier_summary).sort_values(by="Outliers", ascending=False)
    st.dataframe(outlier_df)

    st.markdown("#### Outlier Visualization")
    selected_col = st.selectbox("Select a column to visualize outliers", numerical_cols)
    if selected_col:
        Q1, Q3 = df[selected_col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=selected_col, ax=ax)
        ax.axhline(lower_bound, color='red', linestyle='--', label='Lower Bound')
        ax.axhline(upper_bound, color='blue', linestyle='--', label='Upper Bound')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

# Data Encoder
def encoder(df):
    """
    Encodes categorical columns in the dataset using user-selected methods (Label Encoding,
    One-Hot Encoding, or Frequency Encoding). Provides control over column selection, handles
    missing values, and displays encoding details.
    
    Args:
        df (pd.DataFrame): Input dataset to encode.
    
    Returns:
        pd.DataFrame: Encoded dataset.
    """
    st.subheader("Encode Data")
    
    # Initialize session state for encoded DataFrame
    if 'encoded_df' not in st.session_state:
        st.session_state.encoded_df = df.copy()

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        st.warning("No categorical columns ('object' or 'category') found in the dataset.")
        return df

    # Display original categorical columns
    st.markdown("### Categorical Columns Detected")
    st.write(f"Found {len(categorical_cols)} categorical columns: {', '.join(categorical_cols)}")
    for col in categorical_cols:
        st.write(f"- **{col}**: {df[col].nunique()} unique values, "
                 f"{df[col].isna().sum()} missing ({df[col].isna().sum() / len(df) * 100:.2f}%)")

    # User configuration
    st.markdown("### Encoding Configuration")
    encoding_methods = {
        "Label Encoding": "Assigns integers to categories (best for ordinal data).",
        "One-Hot Encoding": "Creates binary columns for each category (best for non-ordinal data, avoid high cardinality).",
        "Frequency Encoding": "Replaces categories with their frequency (useful for high-cardinality columns)."
    }
    
    # Select columns to encode
    cols_to_encode = st.multiselect("Select Columns to Encode", categorical_cols, default=categorical_cols,
                                   help="Choose which categorical columns to encode. Unselected columns remain unchanged.")
    
    if not cols_to_encode:
        st.warning("Please select at least one column to encode.")
        return st.session_state.encoded_df

    # Missing value handling
    missing_strategy = st.selectbox("Handle Missing Values", 
                                    ["Keep as NaN", "Impute with Mode", "Impute with Custom Value"],
                                    help="Choose how to handle missing values before encoding.")
    custom_value = None
    if missing_strategy == "Impute with Custom Value":
        custom_value = st.text_input("Enter Custom Value for Missing Entries", value="Unknown")

    # Apply missing value handling
    encoded_df = st.session_state.encoded_df.copy()
    for col in cols_to_encode:
        if missing_strategy == "Impute with Mode":
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            encoded_df[col] = df[col].fillna(mode_val)
        elif missing_strategy == "Impute with Custom Value":
            encoded_df[col] = df[col].fillna(custom_value)

    # Encoding method selection per column
    st.markdown("### Assign Encoding Methods")
    encoding_assignments = {}
    for col in cols_to_encode:
        default_method = "One-Hot Encoding" if df[col].nunique() <= 10 else "Frequency Encoding"
        encoding_assignments[col] = st.selectbox(
            f"Encoding Method for {col}",
            list(encoding_methods.keys()),
            index=list(encoding_methods.keys()).index(default_method),
            help=f"{encoding_methods[default_method]} Unique values: {df[col].nunique()}"
        )

    # Apply encoding
    if st.button("Apply Encoding"):
        try:
            for col, method in encoding_assignments.items():
                if method == "Label Encoding":
                    le = LabelEncoder()
                    # Convert to string to handle mixed types and NaNs
                    encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                    st.session_state[f"label_encoder_{col}"] = le  # Store encoder for reference
                    st.write(f"**{col}**: Label Encoded. Classes: {list(le.classes_)}")

                elif method == "One-Hot Encoding":
                    if df[col].nunique() > 50:
                        st.warning(f"**{col}** has {df[col].nunique()} unique values. One-Hot Encoding may create many columns.")
                    # Drop NaN for one-hot encoding, reintroduce after
                    mask = encoded_df[col].notna()
                    ohe_df = pd.get_dummies(encoded_df.loc[mask, col], prefix=col, drop_first=True)
                    encoded_df = pd.concat([encoded_df.drop(columns=[col]), ohe_df], axis=1)
                    encoded_df.loc[~mask, ohe_df.columns] = np.nan
                    st.write(f"**{col}**: One-Hot Encoded. Created {len(ohe_df.columns)} new columns.")

                elif method == "Frequency Encoding":
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    encoded_df[col] = df[col].map(freq_map)
                    st.write(f"**{col}**: Frequency Encoded. Values mapped to proportions.")

            # Update session state
            st.session_state.encoded_df = encoded_df

            # Display results
            st.markdown("### Encoded Dataset Preview")
            st.dataframe(encoded_df.head())
            
            # Data quality check
            new_cols = len(encoded_df.columns) - len(df.columns)
            if new_cols > 0:
                st.info(f"Encoding added {new_cols} new columns.")
            if encoded_df.isna().sum().sum() > 0:
                st.warning(f"Encoded dataset still has {encoded_df.isna().sum().sum()} missing values.")
            
            # Download option
            csv_bytes = encoded_df.to_csv(index=False).encode()
            st.download_button(
                label="Download Encoded Dataset",
                data=csv_bytes,
                file_name="encoded_dataset.csv",
                mime="text/csv",
                key="download_encoded"
            )

        except Exception as e:
            st.error(f"Error during encoding: {str(e)}")
            return df

    # Preview current encoded state
    else:
        st.markdown("### Current Dataset Preview")
        st.dataframe(st.session_state.encoded_df.head())

    return st.session_state.encoded_df

# Data Transformer part eta
def data_transformer(df):
    st.subheader("Data Transformer")
    transformed_df = df.copy()  # Placeholder for future transformations
    st.write("Transformed Dataset:", transformed_df.head())
    return transformed_df

# Data Analysis
def data_analysis(df):
    """
    Performs an in-depth analysis of the dataset, including numerical and categorical summaries,
    interactive visualizations, data quality checks, and column-specific exploration.
    
    Args:
        df (pd.DataFrame): Input dataset to analyze.
    """
    st.subheader("Data Analysis")

    # Initialize tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs(["Summary Statistics", "Visual Exploration", "Data Quality", "Column Deep Dive"])

    with tab1:
        st.markdown("### Summary Statistics")
        # Numerical Columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if numerical_cols.size > 0:
            st.markdown("#### Numerical Columns")
            numerical_summary = df[numerical_cols].describe().T.round(2)
            numerical_summary['Skewness'] = df[numerical_cols].skew().round(2)
            numerical_summary['Kurtosis'] = df[numerical_cols].kurt().round(2)
            numerical_summary['Missing %'] = (df[numerical_cols].isna().sum() / len(df) * 100).round(2)
            st.dataframe(numerical_summary.style.highlight_max(axis=0, color='lightgreen'))
            st.write("*Skewness > 1 or < -1 indicates high skew. Kurtosis > 3 indicates heavy tails.*")
        else:
            st.info("No numerical columns found.")

        # Categorical Columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if categorical_cols.size > 0:
            st.markdown("#### Categorical Columns")
            cat_summary = pd.DataFrame({
                "Column": categorical_cols,
                "Unique Values": [df[col].nunique() for col in categorical_cols],
                "Most Frequent": [df[col].mode()[0] if not df[col].mode().empty else np.nan for col in categorical_cols],
                "Missing %": [(df[col].isna().sum() / len(df) * 100).round(2) for col in categorical_cols]
            })
            st.dataframe(cat_summary)
        else:
            st.info("No categorical columns found.")

    with tab2:
        st.markdown("### Visual Exploration")
        viz_type = st.selectbox("Select Visualization Type", 
                                ["Distribution (Numerical)", "Count Plot (Categorical)", "Correlation Heatmap", "Pair Plot"],
                                key="data_analysis_viz")
        
        if viz_type == "Distribution (Numerical)" and numerical_cols.size > 0:
            col = st.selectbox("Select Column", numerical_cols, key="num_dist_col")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=col, kde=True, bins='auto', ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
            download_image(fig, f"dist_{col}")
            plt.close(fig)

        elif viz_type == "Count Plot (Categorical)" and categorical_cols.size > 0:
            col = st.selectbox("Select Column", categorical_cols, key="cat_count_col")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=col, ax=ax)
            plt.xticks(rotation=45, ha='right')
            ax.set_title(f"Count Plot of {col}")
            st.pyplot(fig)
            download_image(fig, f"count_{col}")
            plt.close(fig)

        elif viz_type == "Correlation Heatmap" and numerical_cols.size > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt='.2f', ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
            download_image(fig, "corr_heatmap")
            plt.close(fig)

        elif viz_type == "Pair Plot" and numerical_cols.size > 0:
            selected_cols = st.multiselect("Select Columns (max 4)", numerical_cols, max_selections=4, key="pair_cols")
            if len(selected_cols) >= 2:
                fig = sns.pairplot(df[selected_cols].dropna())
                st.pyplot(fig)
                img_bytes = BytesIO()
                fig.savefig(img_bytes, format='png', bbox_inches='tight')
                img_bytes.seek(0)
                st.download_button(label="Download Pair Plot", data=img_bytes, 
                                  file_name=f"pairplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                  mime="image/png", key=f"pairplot_{datetime.now().strftime('%H%M%S')}")
                plt.close()

    with tab3:
        st.markdown("### Data Quality Checks")
        # Missing Values
        missing_total = df.isna().sum().sum()
        if missing_total > 0:
            st.warning(f"**Missing Values**: {missing_total} across {df.isna().any().sum()} columns.")
            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing Count": df.isna().sum(),
                "Missing %": (df.isna().sum() / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df["Missing Count"] > 0]
            st.dataframe(missing_df)
        else:
            st.success("No missing values detected.")

        # Duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"**Duplicates**: {duplicates} duplicate rows ({duplicates / len(df) * 100:.2f}%).")
        else:
            st.success("No duplicate rows detected.")

        # Outliers (Numerical)
        if numerical_cols.size > 0:
            outlier_summary = []
            for col in numerical_cols:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    outlier_summary.append({"Column": col, "Outlier Count": outliers, 
                                           "Outlier %": (outliers / len(df) * 100).round(2)})
            if outlier_summary:
                st.warning("**Outliers Detected**:")
                st.dataframe(pd.DataFrame(outlier_summary))
            else:
                st.success("No outliers detected in numerical columns.")

    with tab4:
        st.markdown("### Column Deep Dive")
        selected_col = st.selectbox("Select Column for Detailed Analysis", df.columns, key="deep_dive_col")
        st.write(f"**Column**: {selected_col}")
        st.write(f"**Data Type**: {df[selected_col].dtype}")
        st.write(f"**Missing Values**: {df[selected_col].isna().sum()} ({df[selected_col].isna().sum() / len(df) * 100:.2f}%)")
        st.write(f"**Unique Values**: {df[selected_col].nunique()} ({df[selected_col].nunique() / len(df) * 100:.2f}%)")

        if pd.api.types.is_numeric_dtype(df[selected_col]):
            st.write("**Summary Statistics**:")
            stats = df[selected_col].describe().round(2)
            stats['Skewness'] = df[selected_col].skew().round(2)
            stats['Kurtosis'] = df[selected_col].kurt().round(2)
            st.dataframe(stats)
            fig = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig)
        elif pd.api.types.is_object_dtype(df[selected_col]) or pd.api.types.is_categorical_dtype(df[selected_col]):
            st.write("**Top 5 Values**:")
            value_counts = df[selected_col].value_counts().head(5)
            st.dataframe(pd.DataFrame({
                "Value": value_counts.index,
                "Count": value_counts.values,
                "% of Total": (value_counts.values / len(df) * 100).round(2)
            }))
            fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Top Values in {selected_col}")
            st.plotly_chart(fig)

def download_image(fig, key_prefix):
    """
    Utility function to download a Matplotlib figure as PNG.
    """
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    st.download_button(label="Download Image", data=img_bytes,
                      file_name=f"{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                      mime="image/png", key=f"download_{key_prefix}_{datetime.now().strftime('%H%M%S')}")

# Feature Importance Analysis
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
#Best Parameter Selector
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
# Select ML Models
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, OPTICS
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, 
                             mean_absolute_percentage_error, silhouette_score, davies_bouldin_score, 
                             calinski_harabasz_score, adjusted_rand_score, v_measure_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from io import BytesIO
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress sklearn warnings

# Time series imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from prophet import Prophet
    HAS_STATSMODELS = True
    HAS_PROPHET = True
except ImportError:
    HAS_STATSMODELS = False
    HAS_PROPHET = False

def select_ml_models(df):
    """
    Builds an end-to-end ML workflow for classification, regression, clustering, or time series forecasting.
    Includes preprocessing, model training, evaluation, hyperparameter tuning, and model saving.
    
    Args:
        df (pd.DataFrame): Input dataset as a pandas DataFrame.
    """
    st.subheader("Machine Learning Workflow")

    # Select ML task
    analysis_type = st.selectbox("Select Machine Learning Task", 
                                 ["Classification", "Regression", "Clustering", "Time Series Forecasting"],
                                 help="Choose the type of ML task to perform.")

    # Preprocessing function
    def preprocess_data(df, target_col=None, task_type=None):
        """
        Preprocesses the dataset based on the ML task.
        
        Args:
            df (pd.DataFrame): Input dataset.
            target_col (str): Target column name (None for clustering).
            task_type (str): Type of ML task.
        
        Returns:
            tuple: Preprocessed features (X), target (y), and preprocessor (if applicable).
        """
        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = None

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Define preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        if task_type in ["Classification", "Regression"]:
            # Fit and transform features
            X_processed = preprocessor.fit_transform(X)
            # Handle target for classification
            if task_type == "Classification" and y.dtype in ['object', 'category']:
                le = LabelEncoder()
                y = le.fit_transform(y)
                return X_processed, y, preprocessor, le
            return X_processed, y, preprocessor, None
        elif task_type == "Clustering":
            X_processed = preprocessor.fit_transform(X)
            return X_processed, None, preprocessor, None
        elif task_type == "Time Series Forecasting":
            # Time series requires minimal preprocessing here
            return X, y, None, None

    # Model evaluation function
    def evaluate_model(model, X_test, y_test, task_type, y_pred=None):
        """
        Evaluates the model using task-specific metrics.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test target.
            task_type: Type of ML task.
            y_pred: Predicted values (optional, computed if None).
        
        Returns:
            dict: Evaluation metrics.
        """
        if y_pred is None:
            y_pred = model.predict(X_test) if task_type != "Time Series Forecasting" else model.forecast(len(y_test))

        metrics = {}
        if task_type == "Classification":
            metrics.update({
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "ROC AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], multi_class='ovr') if hasattr(model, "predict_proba") else np.nan
            })
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
            plt.close(fig)

        elif task_type == "Regression":
            metrics.update({
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R²": r2_score(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred) * 100
            })
            # Scatter plot of predictions
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'True Values', 'y': 'Predicted Values'},
                             title="True vs Predicted Values")
            fig.add_scatter(x=y_test, y=y_test, mode='lines', name='Ideal')
            st.plotly_chart(fig)

        elif task_type == "Clustering":
            metrics.update({
                "Silhouette Score": silhouette_score(X_test, y_pred) if len(np.unique(y_pred)) > 1 else np.nan,
                "Davies-Bouldin Index": davies_bouldin_score(X_test, y_pred) if len(np.unique(y_pred)) > 1 else np.nan,
                "Calinski-Harabasz Score": calinski_harabasz_score(X_test, y_pred) if len(np.unique(y_pred)) > 1 else np.nan
            })
            # Visualize clusters (if 2D or reducible)
            if X_test.shape[1] == 2:
                fig = px.scatter(x=X_test[:, 0], y=X_test[:, 1], color=y_pred.astype(str), 
                                 title="Cluster Visualization", labels={'x': 'Feature 1', 'y': 'Feature 2'})
                st.plotly_chart(fig)

        elif task_type == "Time Series Forecasting":
            metrics.update({
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred) * 100
            })
        return metrics

    # Hyperparameter tuning function
    def tune_model(model, X_train, y_train, task_type, model_name):
        """
        Performs hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            model: Model to tune.
            X_train: Training features.
            y_train: Training target.
            task_type: Type of ML task.
            model_name: Name of the model.
        
        Returns:
            tuple: Best model and parameters.
        """
        param_grids = {
            "Logistic Regression": {"C": np.logspace(-3, 3, 10), "penalty": ['l2'], "max_iter": [1000]},
            "Random Forest Classifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
            "SVM Classifier": {"C": [0.1, 1, 10], "kernel": ['rbf', 'linear']},
            "KNN Classifier": {"n_neighbors": [3, 5, 7, 9], "weights": ['uniform', 'distance']},
            "Gradient Boosting Classifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
            "Naive Bayes": {},  # No tuning for Naive Bayes
            "AdaBoost Classifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
            "Extra Trees Classifier": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
            "Linear Regression": {},  # No tuning for basic Linear Regression
            "Ridge Regression": {"alpha": [0.1, 1, 10, 100]},
            "Lasso Regression": {"alpha": [0.1, 1, 10, 100]},
            "Random Forest Regressor": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "SVR": {"C": [0.1, 1, 10], "epsilon": [0.1, 0.2, 0.5]},
            "KNN Regressor": {"n_neighbors": [3, 5, 7, 9], "weights": ['uniform', 'distance']},
            "Gradient Boosting Regressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
            "AdaBoost Regressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
            "Extra Trees Regressor": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
            "K-Means": {"n_clusters": [2, 3, 4, 5, 6, 7, 8]},
            "DBSCAN": {"eps": [0.1, 0.5, 1.0], "min_samples": [3, 5, 10]},
            "Agglomerative Clustering": {"n_clusters": [2, 3, 4, 5, 6, 7, 8]},
            "Spectral Clustering": {"n_clusters": [2, 3, 4, 5, 6, 7, 8]},
            "OPTICS": {"min_samples": [3, 5, 10], "xi": [0.05, 0.1]},
            "ARIMA": {"order": [(1,1,1), (2,1,1), (1,1,2)]},
            "Exponential Smoothing": {"trend": ["add", None], "seasonal": ["add", None]},
            "Prophet": {"changepoint_prior_scale": [0.01, 0.05, 0.1], "seasonality_prior_scale": [5, 10, 15]}
        }

        if model_name in param_grids and param_grids[model_name]:
            search = RandomizedSearchCV(model, param_distributions=param_grids[model_name], 
                                       n_iter=10, cv=3, scoring='accuracy' if task_type == "Classification" else 'r2',
                                       n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
            return search.best_estimator_, search.best_params_
        return model, {}

    # Model saving function
    def save_model(model, model_name):
        """
        Serializes and offers the model for download.
        
        Args:
            model: Trained model.
            model_name: Name of the model.
        
        Returns:
            BytesIO: Serialized model file.
        """
        model_file = BytesIO()
        pickle.dump(model, model_file)
        model_file.seek(0)
        return model_file

    # Classification and Regression
    if analysis_type in ["Classification", "Regression"]:
        st.markdown("### Configure Model")
        target_col = st.selectbox("Select Target Variable", df.columns, key="target_col")
        feature_cols = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_col],
                                      key="feature_cols")
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return

        # Validate target
        if analysis_type == "Classification":
            if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > len(df) // 10:
                st.error("Target appears continuous. Consider binning or switching to Regression.")
                return
        elif analysis_type == "Regression":
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                st.error("Target must be numeric for regression.")
                return

        # Preprocess data
        try:
            X, y, preprocessor, le = preprocess_data(df[feature_cols + [target_col]], target_col, analysis_type)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            st.error(f"Preprocessing error: {str(e)}")
            return

        # Model options
        model_options = {
            "Classification": {
                "Logistic Regression": LogisticRegression(random_state=42),
                "Random Forest Classifier": RandomForestClassifier(random_state=42),
                "SVM Classifier": SVC(random_state=42, probability=True),
                "KNN Classifier": KNeighborsClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
                "Naive Bayes": GaussianNB(),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "Extra Trees Classifier": ExtraTreesClassifier(random_state=42)
            },
            "Regression": {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(random_state=42),
                "Lasso Regression": Lasso(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "SVR": SVR(),
                "KNN Regressor": KNeighborsRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                "Extra Trees Regressor": ExtraTreesRegressor(random_state=42)
            }
        }[analysis_type]

        selected_model_name = st.selectbox("Select Model", list(model_options.keys()), key="model_select")
        model = model_options[selected_model_name]

        # Train model
        if st.button("Train Model", key="train_button"):
            with st.spinner("Training model..."):
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = evaluate_model(model, X_test, y_test, analysis_type, y_pred)
                    st.markdown("### Model Performance")
                    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).round(4))

                    # Cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=5, 
                                                scoring='accuracy' if analysis_type == "Classification" else 'r2')
                    st.write(f"**Cross-Validation Scores** (5-fold): Mean = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")

                    # Store model in session state
                    st.session_state['trained_model'] = model
                    st.session_state['model_name'] = selected_model_name

                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    return

        # Hyperparameter tuning
        if st.button("Perform Hyperparameter Tuning", key="tune_button"):
            with st.spinner("Tuning hyperparameters..."):
                try:
                    tuned_model, best_params = tune_model(model, X_train, y_train, analysis_type, selected_model_name)
                    tuned_model.fit(X_train, y_train)
                    y_pred = tuned_model.predict(X_test)
                    metrics = evaluate_model(tuned_model, X_test, y_test, analysis_type, y_pred)
                    st.markdown("### Tuned Model Performance")
                    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).round(4))
                    st.write("**Best Hyperparameters**:", best_params)

                    # Update stored model
                    st.session_state['trained_model'] = tuned_model
                    st.session_state['model_name'] = selected_model_name + "_Tuned"

                except Exception as e:
                    st.error(f"Tuning error: {str(e)}")

        # Save model
        if 'trained_model' in st.session_state and st.button("Save The Model", key="save_button"):
            model_file = save_model(st.session_state['trained_model'], st.session_state['model_name'])
            st.download_button(
                label="Download Model",
                data=model_file,
                file_name=f"{st.session_state['model_name'].replace(' ', '_').lower()}_model.pkl",
                mime="application/octet-stream",
                key="download_model"
            )

    # Clustering
    elif analysis_type == "Clustering":
        st.markdown("### Configure Clustering")
        feature_cols = st.multiselect("Select Features for Clustering", df.columns, key="cluster_cols")
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return

        # Preprocess data
        try:
            X, _, preprocessor, _ = preprocess_data(df[feature_cols], task_type="Clustering")
        except Exception as e:
            st.error(f"Preprocessing error: {str(e)}")
            return

        n_clusters = st.slider("Number of Clusters (for applicable algorithms)", 2, 10, 3, key="n_clusters")
        clustering_models = {
            "K-Means": KMeans(n_clusters=n_clusters, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
            "Agglomerative Clustering": AgglomerativeClustering(n_clusters=n_clusters),
            "Spectral Clustering": SpectralClustering(n_clusters=n_clusters, random_state=42),
            "OPTICS": OPTICS(min_samples=5)
        }

        selected_model_name = st.selectbox("Select Clustering Algorithm", list(clustering_models.keys()), 
                                           key="cluster_model_select")
        model = clustering_models[selected_model_name]

        # Train model
        if st.button("Perform Clustering", key="cluster_button"):
            with st.spinner("Performing clustering..."):
                try:
                    clusters = model.fit_predict(X)
                    df_with_clusters = df.copy()
                    df_with_clusters['Cluster'] = clusters
                    st.markdown("### Clustered Data Sample")
                    st.dataframe(df_with_clusters.head())

                    # Evaluate clustering
                    metrics = evaluate_model(model, X, clusters, "Clustering", clusters)
                    st.markdown("### Clustering Metrics")
                    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).round(4))

                    # Store model
                    st.session_state['trained_model'] = model
                    st.session_state['model_name'] = selected_model_name

                except Exception as e:
                    st.error(f"Clustering error: {str(e)}")
                    return

        # Hyperparameter tuning
        if st.button("Perform Hyperparameter Tuning", key="cluster_tune_button"):
            with st.spinner("Tuning hyperparameters..."):
                try:
                    tuned_model, best_params = tune_model(model, X, clusters, "Clustering", selected_model_name)
                    clusters = tuned_model.fit_predict(X)
                    df_with_clusters = df.copy()
                    df_with_clusters['Cluster'] = clusters
                    st.markdown("### Tuned Clustered Data Sample")
                    st.dataframe(df_with_clusters.head())

                    metrics = evaluate_model(tuned_model, X, clusters, "Clustering", clusters)
                    st.markdown("### Tuned Clustering Metrics")
                    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).round(4))
                    st.write("**Best Hyperparameters**:", best_params)

                    # Update stored model
                    st.session_state['trained_model'] = tuned_model
                    st.session_state['model_name'] = selected_model_name + "_Tuned"

                except Exception as e:
                    st.error(f"Tuning error: {str(e)}")

        # Save model
        if 'trained_model' in st.session_state and st.button("Save The Model", key="cluster_save_button"):
            model_file = save_model(st.session_state['trained_model'], st.session_state['model_name'])
            st.download_button(
                label="Download Model",
                data=model_file,
                file_name=f"{st.session_state['model_name'].replace(' ', '_').lower()}_model.pkl",
                mime="application/octet-stream",
                key="download_cluster_model"
            )

    # Time Series Forecasting
    elif analysis_type == "Time Series Forecasting":
        if not (HAS_STATSMODELS or HAS_PROPHET):
            st.error("Please install statsmodels or prophet: `pip install statsmodels fbprophet`")
            return

        st.markdown("### Configure Time Series Forecasting")
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if datetime_cols.empty:
            st.error("No datetime columns found. Please ensure a datetime column exists.")
            return

        date_col = st.selectbox("Select Date Column", datetime_cols, key="date_col")
        value_col = st.selectbox("Select Value Column", 
                                 df.select_dtypes(include=['float64', 'int64']).columns, 
                                 key="value_col")

        # Prepare data
        try:
            ts_df = df[[date_col, value_col]].sort_values(date_col).dropna()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        except Exception as e:
            st.error(f"Data preparation error: {str(e)}")
            return

        # Time-based split
        train_size = st.slider("Training Data Proportion (%)", 50, 95, 80, key="train_size")
        train_size = int(len(ts_df) * (train_size / 100))
        train, test = ts_df[:train_size], ts_df[train_size:]

        # Model options
        forecast_models = {}
        if HAS_STATSMODELS:
            forecast_models.update({
                "ARIMA": lambda data: ARIMA(data[value_col], order=(1,1,1)).fit(),
                "Exponential Smoothing": lambda data: ExponentialSmoothing(
                    data[value_col], trend='add', seasonal='add', seasonal_periods=12
                ).fit()
            })
        if HAS_PROPHET:
            forecast_models["Prophet"] = lambda data: Prophet().fit(
                data.rename(columns={date_col: 'ds', value_col: 'y'})
            )

        selected_model_name = st.selectbox("Select Forecasting Model", list(forecast_models.keys()), 
                                           key="ts_model_select")

        # Train model
        if st.button("Train Model", key="ts_train_button"):
            with st.spinner("Training time series model..."):
                try:
                    if selected_model_name == "Prophet":
                        model = forecast_models[selected_model_name](train)
                        future = model.make_future_dataframe(periods=len(test))
                        forecast = model.predict(future)
                        y_pred = forecast['yhat'][-len(test):].values
                        y_test = test[value_col].values
                    else:
                        model = forecast_models[selected_model_name](train)
                        y_pred = model.forecast(steps=len(test))
                        y_test = test[value_col].values

                    # Evaluate
                    metrics = evaluate_model(model, test, y_test, "Time Series Forecasting", y_pred)
                    st.markdown("### Model Performance")
                    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).round(4))

                    # Plot forecast
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(train[date_col], train[value_col], label="Train")
                    ax.plot(test[date_col], test[value_col], label="Test")
                    ax.plot(test[date_col], y_pred, label="Forecast")
                    ax.set_title(f"{selected_model_name} Forecast")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Store model
                    st.session_state['trained_model'] = model
                    st.session_state['model_name'] = selected_model_name

                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    return

        # Hyperparameter tuning
        if st.button("Perform Hyperparameter Tuning", key="ts_tune_button"):
            with st.spinner("Tuning hyperparameters..."):
                try:
                    if selected_model_name == "Prophet":
                        param_grid = {
                            "changepoint_prior_scale": [0.01, 0.05, 0.1],
                            "seasonality_prior_scale": [5, 10, 15]
                        }
                        best_score = float('inf')
                        best_params = {}
                        best_model = None
                        for cps in param_grid["changepoint_prior_scale"]:
                            for sps in param_grid["seasonality_prior_scale"]:
                                model = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps)
                                model.fit(train.rename(columns={date_col: 'ds', value_col: 'y'}))
                                future = model.make_future_dataframe(periods=len(test))
                                forecast = model.predict(future)
                                score = mean_squared_error(test[value_col], forecast['yhat'][-len(test):])
                                if score < best_score:
                                    best_score = score
                                    best_params = {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps}
                                    best_model = model
                        model = best_model
                    else:
                        model, best_params = tune_model(model, train[value_col], None, 
                                                        "Time Series Forecasting", selected_model_name)

                    # Re-evaluate
                    if selected_model_name == "Prophet":
                        future = model.make_future_dataframe(periods=len(test))
                        forecast = model.predict(future)
                        y_pred = forecast['yhat'][-len(test):].values
                        y_test = test[value_col].values
                    else:
                        model = forecast_models[selected_model_name](train)
                        y_pred = model.forecast(steps=len(test))
                        y_test = test[value_col].values

                    metrics = evaluate_model(model, test, y_test, "Time Series Forecasting", y_pred)
                    st.markdown("### Tuned Model Performance")
                    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).round(4))
                    st.write("**Best Hyperparameters**:", best_params)

                    # Plot tuned forecast
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(train[date_col], train[value_col], label="Train")
                    ax.plot(test[date_col], test[value_col], label="Test")
                    ax.plot(test[date_col], y_pred, label="Tuned Forecast")
                    ax.set_title(f"Tuned {selected_model_name} Forecast")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Update stored model
                    st.session_state['trained_model'] = model
                    st.session_state['model_name'] = selected_model_name + "_Tuned"

                except Exception as e:
                    st.error(f"Tuning error: {str(e)}")

        # Save model
        if 'trained_model' in st.session_state and st.button("Save The Model", key="ts_save_button"):
            model_file = save_model(st.session_state['trained_model'], st.session_state['model_name'])
            st.download_button(
                label="Download Model",
                data=model_file,
                file_name=f"{st.session_state['model_name'].replace(' ', '_').lower()}_model.pkl",
                mime="application/octet-stream",
                key="download_ts_model"
            )

# Clear Modified Dataset
def clear_modified_dataset():
    st.subheader("Clear Modified Dataset")
    st.session_state.pop('uploaded_df', None)
    st.write("Dataset cleared.")

# Chat with the dataset
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
# NLP Pipeline

def nlp_pipeline_tab():
    """Function for the NLP Pipeline tab."""
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                roc_auc_score, confusion_matrix)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from gensim import corpora
    from gensim.models import LdaModel
    import spacy
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from io import BytesIO
    import pickle

    st.header("NLP Pipeline")
    st.markdown("Perform Natural Language Processing tasks like text classification, sentiment analysis, topic modeling, and named entity recognition.")

    # Access uploaded dataset from session state
    if 'uploaded_df' not in st.session_state:
        st.error("No dataset uploaded. Please upload a CSV file in the main DataGenie app.")
        return

    df = st.session_state['uploaded_df']
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Select NLP task
    nlp_task = st.selectbox("Select NLP Task", 
                            ["Text Classification", "Sentiment Analysis", "Topic Modeling", "Named Entity Recognition"],
                            help="Choose the specific NLP task to perform.")

    # Preprocessing function
    def preprocess_nlp_data(df, text_col, target_col=None, task_type=None):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        def clean_text(text):
            if not isinstance(text, str):
                return ""
            tokens = word_tokenize(text.lower())
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
            tokens = [token for token in tokens if token not in stop_words]
            return " ".join(tokens)

        X_text = df[text_col].apply(clean_text)

        if task_type in ["Text Classification", "Sentiment Analysis"]:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = vectorizer.fit_transform(X_text).toarray()
            y = df[target_col] if target_col else None
            if task_type == "Text Classification" and y.dtype in ['object', 'category']:
                le = LabelEncoder()
                y = le.fit_transform(y)
                return X, y, vectorizer, le
            return X, y, vectorizer, None
        elif task_type == "Topic Modeling":
            texts = [text.split() for text in X_text]
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            return corpus, None, dictionary, None
        elif task_type == "Named Entity Recognition":
            return X_text, None, None, None

    # Evaluation function
    def evaluate_model(model, X_test, y_test, task_type, y_pred=None, vectorizer=None):
        metrics = {}
        if task_type in ["Text Classification", "Sentiment Analysis"]:
            if y_pred is None:
                y_pred = model.predict(X_test)
            metrics.update({
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "ROC AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], multi_class='ovr') if hasattr(model, "predict_proba") else np.nan
            })
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
            plt.close(fig)
        elif task_type == "Topic Modeling":
            metrics["Number of Topics"] = model.num_topics
            topics = model.print_topics(num_words=5)
            for topic_id, topic in topics:
                st.write(f"**Topic {topic_id}**: {topic}")
        return metrics

    # Hyperparameter tuning function
    def tune_model(model, X_train, y_train, task_type, model_name):
        param_grids = {
            "Logistic Regression": {"C": np.logspace(-3, 3, 10), "penalty": ['l2'], "max_iter": [1000]},
            "Random Forest Classifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "SVM Classifier": {"C": [0.1, 1, 10], "kernel": ['rbf', 'linear']},
            "Gradient Boosting Classifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
            "Naive Bayes": {"alpha": [0.1, 0.5, 1.0]},
            "LDA": {"num_topics": [5, 10, 15, 20]}
        }
        if model_name in param_grids and param_grids[model_name]:
            search = RandomizedSearchCV(model, param_distributions=param_grids[model_name], 
                                       n_iter=10, cv=3, scoring='accuracy' if task_type in ["Text Classification", "Sentiment Analysis"] else None,
                                       n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
            return search.best_estimator_, search.best_params_
        return model, {}

    # Model saving function
    def save_model(model, model_name):
        model_file = BytesIO()
        pickle.dump(model, model_file)
        model_file.seek(0)
        return model_file

    # Identify text columns
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not text_cols:
        st.error("No text columns found in the dataset.")
        return
    text_col = st.selectbox("Select Text Column", text_cols, key="nlp_text_col")

    # Text Classification and Sentiment Analysis
    if nlp_task in ["Text Classification", "Sentiment Analysis"]:
        target_col = st.selectbox("Select Target Variable", 
                                  [col for col in df.columns if col != text_col], 
                                  key="nlp_target_col")
        if nlp_task == "Sentiment Analysis" and df[target_col].nunique() > 10:
            st.error("Target has too many unique values for sentiment analysis.")
            return

        try:
            X, y, vectorizer, le = preprocess_nlp_data(df, text_col, target_col, nlp_task)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            st.error(f"Preprocessing error: {str(e)}")
            return

        model_options = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(random_state=42),
            "SVM Classifier": SVC(random_state=42, probability=True),
            "Naive Bayes": MultinomialNB(),
            "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)
        }
        selected_model_name = st.selectbox("Select Model", list(model_options.keys()), key="nlp_model_select")
        model = model_options[selected_model_name]

        if st.button("Train Model", key="nlp_train_button"):
            with st.spinner("Training NLP model..."):
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = evaluate_model(model, X_test, y_test, nlp_task, y_pred, vectorizer)
                    st.markdown("### Model Performance")
                    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).round(4))

                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                    st.write(f"**Cross-Validation Scores** (5-fold): Mean = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")

                    if selected_model_name in ["Random Forest Classifier", "Gradient Boosting Classifier"]:
                        feature_names = vectorizer.get_feature_names_out()
                        importance = model.feature_importances_
                        top_features = pd.DataFrame({"Feature": feature_names, "Importance": importance}).nlargest(10, "Importance")
                        fig = px.bar(top_features, x="Feature", y="Importance", title="Top 10 Feature Importances")
                        st.plotly_chart(fig)

                    st.session_state['nlp_trained_model'] = model
                    st.session_state['nlp_model_name'] = selected_model_name
                    st.session_state['vectorizer'] = vectorizer
                    st.session_state['label_encoder'] = le

                except Exception as e:
                    st.error(f"Training error: {str(e)}")

        if st.button("Perform Hyperparameter Tuning", key="nlp_tune_button"):
            with st.spinner("Tuning hyperparameters..."):
                try:
                    tuned_model, best_params = tune_model(model, X_train, y_train, nlp_task, selected_model_name)
                    tuned_model.fit(X_train, y_train)
                    y_pred = tuned_model.predict(X_test)
                    metrics = evaluate_model(tuned_model, X_test, y_test, nlp_task, y_pred, vectorizer)
                    st.markdown("### Tuned Model Performance")
                    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).round(4))
                    st.write("**Best Hyperparameters**:", best_params)

                    st.session_state['nlp_trained_model'] = tuned_model
                    st.session_state['nlp_model_name'] = selected_model_name + "_Tuned"

                except Exception as e:
                    st.error(f"Tuning error: {str(e)}")

        if 'nlp_trained_model' in st.session_state and st.button("Save The Model", key="nlp_save_button"):
            model_file = save_model(st.session_state['nlp_trained_model'], st.session_state['nlp_model_name'])
            st.download_button(
                label="Download Model",
                data=model_file,
                file_name=f"{st.session_state['nlp_model_name'].replace(' ', '_').lower()}_model.pkl",
                mime="application/octet-stream",
                key="download_nlp_model"
            )

    # Topic Modeling
    elif nlp_task == "Topic Modeling":
        try:
            corpus, _, dictionary, _ = preprocess_nlp_data(df, text_col, task_type="Topic Modeling")
        except Exception as e:
            st.error(f"Preprocessing error: {str(e)}")
            return

        num_topics = st.slider("Number of Topics", 2, 20, 5, key="num_topics")
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

        if st.button("Perform Topic Modeling", key="topic_button"):
            with st.spinner("Performing topic modeling..."):
                try:
                    metrics = evaluate_model(model, corpus, None, "Topic Modeling")
                    st.markdown("### Topic Modeling Results")
                    st.write("**Topics Identified**:")
                    for topic_id, topic in metrics.items():
                        if isinstance(topic, str):
                            st.write(f"- {topic}")

                    st.session_state['nlp_trained_model'] = model
                    st.session_state['nlp_model_name'] = "LDA"

                except Exception as e:
                    st.error(f"Topic modeling error: {str(e)}")

        if st.button("Perform Hyperparameter Tuning", key="topic_tune_button"):
            with st.spinner("Tuning topics..."):
                try:
                    tuned_model, best_params = tune_model(model, corpus, None, "Topic Modeling", "LDA")
                    metrics = evaluate_model(tuned_model, corpus, None, "Topic Modeling")
                    st.markdown("### Tuned Topic Modeling Results")
                    st.write("**Tuned Topics Identified**:")
                    for topic_id, topic in metrics.items():
                        if isinstance(topic, str):
                            st.write(f"- {topic}")
                    st.write("**Best Hyperparameters**:", best_params)

                    st.session_state['nlp_trained_model'] = tuned_model
                    st.session_state['nlp_model_name'] = "LDA_Tuned"

                except Exception as e:
                    st.error(f"Tuning error: {str(e)}")

        if 'nlp_trained_model' in st.session_state and st.button("Save The Model", key="topic_save_button"):
            model_file = save_model(st.session_state['nlp_trained_model'], st.session_state['nlp_model_name'])
            st.download_button(
                label="Download Model",
                data=model_file,
                file_name=f"{st.session_state['nlp_model_name'].replace(' ', '_').lower()}_model.pkl",
                mime="application/octet-stream",
                key="download_topic_model"
            )

    # Named Entity Recognition
    elif nlp_task == "Named Entity Recognition":
        try:
            X_text, _, _, _ = preprocess_nlp_data(df, text_col, task_type="Named Entity Recognition")
        except Exception as e:
            st.error(f"Preprocessing error: {str(e)}")
            return

        nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat"])
        if st.button("Perform NER", key="ner_button"):
            with st.spinner("Performing Named Entity Recognition..."):
                try:
                    st.markdown("### NER Results")
                    for text in X_text[:10]:
                        doc = nlp(text)
                        st.write(f"**Text**: {text[:100]}...")
                        entities = [(ent.text, ent.label_) for ent in doc.ents]
                        if entities:
                            st.dataframe(pd.DataFrame(entities, columns=["Entity", "Label"]))
                        else:
                            st.write("No entities detected.")
                    st.info("NER uses a pre-trained spaCy model, so no training or saving is required.")
                except Exception as e:
                    st.error(f"NER error: {str(e)}")

# Main app layout
add_custom_styles()
st.title("")
add_header()

tab1, tab2, tab3, tab4 = st.tabs(["Dataset Generator", "Example Prompts", "Chat with Dataset", "NLP Pipeline"])

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

# NLP Pipeline Tab
with tab4:
    nlp_pipeline_tab()

# Line ~1100: Footer
add_footer()

# Sidebar for data processing and visualization
add_sidebar()
feature_options = st.sidebar.radio("Select Option", ["Dataset Overview", "Clean Data", "Detect Outlier", "Encoder",
                                                    "Data Transformer", "Data Analysis", "Feature Importance Analyzer",
                                                    "Best Parameter Selector", "Train The Dataset", "Clear Modified Dataset",
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
        elif feature_options == "Train The Dataset":
            select_ml_models(df)
        elif feature_options == "Clear Modified Dataset":
            clear_modified_dataset()
        elif feature_options == "Visualizations":
            visualize_dataset(df)
            features = st.sidebar.multiselect("Select features for specific visualizations", df.columns.tolist())
            if features:
                visualize_specific_features(df, features)
        elif feature_options == "NLP Pipeline":
            nlp_pipeline_tab()

        if 'uploaded_df' in st.session_state:
            df = st.session_state['uploaded_df']
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
else:
    st.sidebar.info("Upload a CSV to proceed.")
