# DataGenie

**DataGenie** is an AI-powered, Streamlit-based data science assistant designed to simplify and accelerate data-related workflows. Built with Python, it integrates powerful libraries like `pandas`, `scikit-learn`, `seaborn`, `plotly`, and `spacy`, along with the Groq API for AI-driven capabilities. DataGenie empowers users to generate synthetic datasets, preprocess data, perform exploratory data analysis (EDA), train machine learning models, execute natural language processing (NLP) tasks, and query datasets interactively using natural language. Whether you're a beginner exploring data science or an experienced practitioner prototyping solutions, DataGenie is your go-to tool for end-to-end data workflows.

Inspired by the project "Predicta" by Ahammad Nafiz, DataGenie combines versatility, interactivity, and AI to make data science accessible and efficient.

## Table of Contents

- Features
- Tech Stack
- Installation
- Usage
- Project Structure
- Detailed Features
  - Synthetic Dataset Generation
  - Data Preprocessing
  - Exploratory Data Analysis (EDA)
  - Machine Learning Workflow
  - NLP Pipeline
  - Chat with Dataset
  - Additional Utilities
- Contributing

## Features

DataGenie offers a comprehensive suite of tools for data science and machine learning:

- **Synthetic Dataset Generation**: Create realistic datasets using natural language prompts, powered by the Groq API.
- **Data Preprocessing**: Clean datasets, encode categorical variables, detect outliers, and transform data.
- **Exploratory Data Analysis (EDA)**: Generate summaries, visualizations, and data quality reports.
- **Machine Learning**: Train, evaluate, and tune models for classification, regression, clustering, and time series forecasting.
- **Natural Language Processing (NLP)**: Perform text classification, sentiment analysis, topic modeling, and named entity recognition (NER).
- **Chat with Dataset**: Query datasets using natural language for insights and custom analyses.
- **Interactive UI**: Streamlit-based interface with tabs, sidebars, and downloadable outputs (CSV, images, models).
- **Custom Visualizations**: Create histograms, scatter plots, heatmaps, and more with downloadable PNGs.
- **Model Saving**: Export trained ML and NLP models as `.pkl` files.
- **Error Handling**: Robust validation and user-friendly error messages for reliable operation.

## Tech Stack

DataGenie is built with the following technologies:

- **Python**: Core programming language.
- **Streamlit**: Web framework for the interactive UI.
- **Pandas & NumPy**: Data manipulation and analysis.
- **Faker**: Synthetic data generation.
- **Scikit-learn**: Machine learning models and preprocessing.
- **Matplotlib, Seaborn, Plotly**: Data visualization.
- **Spacy, NLTK, Gensim**: NLP tasks.
- **Statsmodels, Prophet** (optional): Time series forecasting.
- **Groq API**: AI-driven code generation and dataset querying.
- **CSS**: Custom styling with Google Fonts (`Roboto`).

## Installation

To run DataGenie locally, follow these steps:

### Prerequisites

- Python 3.8 or higher.
- Git for cloning the repository.
- A Groq API key (sign up at xAI to obtain one).

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Mahatir-Ahmed-Tusher/DataGenie.git
   cd DataGenie
   ```

2. **Create a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**: Install the required packages using the provided `requirements.txt` (or create one based on the list below):

   ```bash
   pip install streamlit pandas numpy faker groq matplotlib seaborn plotly scikit-learn spacy nltk gensim
   ```

   Optional dependencies for time series forecasting:

   ```bash
   pip install statsmodels prophet
   ```

4. **Download NLTK Resources**: The app automatically downloads required NLTK resources (`punkt_tab`, `stopwords`, `wordnet`) on first run. Ensure an internet connection or pre-download them:

   ```python
   import nltk
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

5. **Install spaCy Model**: The NLP pipeline requires the `en_core_web_sm` model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Set Up Groq API Key**: Replace the hardcoded API key in `app.py` (line \~300) with your own:

   ```python
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

   Alternatively, set it as an environment variable:

   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"  # On Windows: set GROQ_API_KEY=your_groq_api_key_here
   ```

7. **Run the Application**: Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   Open your browser to `http://localhost:8501` to access DataGenie.

## Usage

1. **Launch the App**: Run `streamlit run app.py` and navigate to the provided URL.
2. **Upload a Dataset**: Use the file uploader in the header to upload a CSV file for analysis.
3. **Generate Synthetic Data**: Go to the "Dataset Generator" tab, enter a prompt (e.g., "Generate 500 rows of real estate data with property type, location, size (sq ft), price, and status (Available/Sold)"), and download the resulting CSV.
4. **Preprocess Data**: Use the sidebar to clean data, encode variables, or detect outliers.
5. **Perform EDA**: Explore dataset summaries and visualizations via the "Data Analysis" or "Visualizations" options.
6. **Train ML Models**: Select "Train The Dataset" to configure and train classification, regression, clustering, or time series models.
7. **Run NLP Tasks**: Use the "NLP Pipeline" tab for text classification, sentiment analysis, topic modeling, or NER.
8. **Chat with Dataset**: Upload a CSV in the "Chat with Dataset" tab and ask questions like "What is the average value of column X?"
9. **Download Outputs**: Save datasets, visualizations, or trained models as CSV, PNG, or `.pkl` files.

## Project Structure

```
DataGenie/
├── app.py              # Main Streamlit application
├── README.md           # Project documentation (this file)
├── requirements.txt    # Dependencies (create manually if needed)
└── nltk_data/          # NLTK resources (downloaded automatically)
```

- **app.py**: Contains all logic, including UI, data processing, ML, NLP, and API integrations.
- **requirements.txt**: Should include `streamlit`, `pandas`, `numpy`, `faker`, `groq`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `spacy`, `nltk`, `gensim`, and optional `statsmodels`, `prophet`.

## Detailed Features

### Synthetic Dataset Generation

- **Prompt-Based Generation**: Users input natural language prompts to create datasets with specified columns, row counts, and data types.
- **Groq API Integration**: Generates Python code using `pandas`, `faker`, and `random` to produce realistic data.
- **Validation**: Ensures correct row counts and column consistency.
- **Downloadable Output**: Datasets can be downloaded as CSV files.
- **Example Prompts**: Provided for domains like Finance, Healthcare, Education, and Real Estate (e.g., "Generate 1000 patient records with age, gender, blood pressure, cholesterol level, and diagnosis").

### Data Preprocessing

- **Dataset Overview**:
  - Displays size, memory usage, data types, missing values, and duplicates.
  - Summarizes numerical (mean, std, skewness) and categorical (top values) columns.
- **Clean Data**:
  - Removes missing values and duplicates.
  - Shows missing value statistics.
- **Detect Outlier**:
  - Uses IQR method for numerical columns.
  - Visualizes outliers with box plots.
- **Encoder**:
  - Supports Label Encoding, One-Hot Encoding, and Frequency Encoding.
  - Handles missing values (NaN, mode, custom value).
  - Downloads encoded datasets.
- **Data Transformer**:
  - Placeholder for future transformations (currently returns input DataFrame).

### Exploratory Data Analysis (EDA)

- **Data Analysis**:
  - **Summary Statistics**: Numerical and categorical summaries with skewness and kurtosis.
  - **Visual Exploration**: Histograms, count plots, correlation heatmaps, pair plots.
  - **Data Quality**: Reports missing values, duplicates, and outliers.
  - **Column Deep Dive**: Detailed stats and visualizations for a selected column.
- **Visualizations**:
  - Supports Histogram, Box Plot, Scatter Plot, Count Plot, Correlation Heatmap, and Time Series.
  - Interactive column selection.
  - Downloadable PNG outputs.

### Machine Learning Workflow

- **Tasks Supported**:
  - **Classification**: Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting, Naive Bayes, AdaBoost, Extra Trees.
    - Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC.
    - Visualizations: Confusion Matrix.
  - **Regression**: Linear Regression, Ridge, Lasso, Random Forest, SVR, KNN, Gradient Boosting, AdaBoost, Extra Trees.
    - Metrics: MAE, MSE, RMSE, R², MAPE.
    - Visualizations: True vs Predicted Scatter Plot.
  - **Clustering**: K-Means, DBSCAN, Agglomerative, Spectral, OPTICS.
    - Metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score.
    - Visualizations: Cluster Scatter Plot (if 2D).
  - **Time Series Forecasting**: ARIMA, Exponential Smoothing, Prophet (optional).
    - Metrics: MAE, MSE, RMSE, MAPE.
    - Visualizations: Train/Test/Forecast Line Plot.
- **Features**:
  - Preprocessing: Imputation, scaling, one-hot encoding.
  - Train-test split (80-20).
  - Cross-validation (5-fold).
  - Hyperparameter tuning with `RandomizedSearchCV`.
  - Model saving as `.pkl` files.

### NLP Pipeline

- **Tasks Supported**:
  - **Text Classification**: Logistic Regression, Random Forest, SVM, Naive Bayes, Gradient Boosting.
    - Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC.
    - Visualizations: Confusion Matrix, Feature Importances.
  - **Sentiment Analysis**: Similar to Text Classification with target validation.
  - **Topic Modeling**: LDA with customizable topic counts.
    - Outputs top words per topic.
  - **Named Entity Recognition (NER)**: Uses spaCy’s `en_core_web_sm` model.
    - Displays entities and labels.
- **Features**:
  - Preprocessing: Text cleaning, tokenization, stopword removal, lemmatization, TF-IDF vectorization.
  - Cross-validation and hyperparameter tuning.
  - Model saving as `.pkl` files.

### Chat with Dataset

- **Functionality**: Query uploaded CSVs using natural language (e.g., "Show the top 5 rows").
- **Groq API**: Generates and executes pandas-based Python code.
- **Outputs**: Displays results, DataFrames, or visualizations.
- **Safety**: Executes code in a restricted environment.

### Additional Utilities

- **Feature Importance Analyzer**: Uses Random Forest to rank feature importances with bar plots.
- **Best Parameter Selector**: Tunes hyperparameters for classification and regression models using `GridSearchCV`.
- **Clear Modified Dataset**: Resets the uploaded dataset from session state.

## Contributing

DataGenie is an open-source project, and we welcome contributions from the community! Whether you’re fixing bugs, adding new features, improving documentation, or suggesting enhancements, your input is valuable. Here’s how you can contribute:

1. **Fork the Repository**: Create your own copy of the project.
2. **Create a Branch**: Work on a feature or bug fix in a dedicated branch (`git checkout -b feature-name`).
3. **Make Changes**: Implement your changes with clear, well-documented code.
4. **Test Thoroughly**: Ensure your changes don’t break existing functionality.
5. **Submit a Pull Request**: Describe your changes in detail and link to any relevant issues.
6. **Follow Guidelines**: Adhere to Python PEP 8 style and include tests if possible.

Ideas for contributions:

- Add new ML models (e.g., deep learning with TensorFlow).
- Enhance NLP tasks (e.g., text summarization, question answering).
- Improve performance with caching or asynchronous processing.
- Create a `requirements.txt` file or Docker setup.
- Add user guides or tutorials.

Join us in making DataGenie even better! Check out the Issues page for open tasks or propose your own ideas.

---

**Developed by**: Mahatir Ahmed Tusher\
**Inspired by**: Predicta by Ahammad Nafiz