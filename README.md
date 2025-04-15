
# DataGenie

![DataGenie Logo](https://i.postimg.cc/5y20B10S/89c59ca6-c8a8-4210-ba7b-f77a44a8fa3a-removalai-preview.png)

**DataGenie** is an AI-powered Streamlit application designed to simplify your data science workflow. It enables users to generate synthetic datasets, preprocess data, perform exploratory data analysis (EDA), train machine learning models, and interact with datasets using natural language queries. Powered by the Groq API, Faker, and a suite of ML and visualization libraries, DataGenie is your go-to tool for data exploration and modeling.

🚀 **Live Demo**: [Try DataGenie on Hugging Face Spaces](https://huggingface.co/spaces/MahatirTusher/DataGenie)

---

## Features

DataGenie offers a comprehensive set of tools for data scientists, analysts, and enthusiasts:

- **Synthetic Dataset Generation**:
  - Create realistic datasets using natural language prompts (e.g., "Generate 500 rows of customer data with name, age, and purchase amount").
  - Powered by Faker and Groq API for dynamic, domain-specific data.
  - Download generated datasets as CSV files.

- **Data Preprocessing**:
  - Clean datasets by removing duplicates and missing values.
  - Detect outliers using the Interquartile Range (IQR) method.
  - Encode categorical variables with LabelEncoder.
  - Placeholder for advanced transformations (e.g., scaling, PCA).

- **Exploratory Data Analysis (EDA)**:
  - Visualize data with histograms, box plots, scatter plots, count plots, correlation heatmaps, and time series plots.
  - Supports Matplotlib, Seaborn, and Plotly for static and interactive visualizations.
  - Download visualizations as PNG files.

- **Machine Learning**:
  - Train models for classification (Logistic Regression, Random Forest, SVM, KNN), regression (Linear Regression, Random Forest, SVR, Decision Tree), clustering (K-Means, DBSCAN, Agglomerative), and time series forecasting (Exponential Smoothing, ARIMA).
  - Perform hyperparameter tuning with GridSearchCV.
  - Analyze feature importance using Random Forest.
  - Evaluate models with metrics like accuracy, F1 score, R², and more.

- **Chat with Dataset**:
  - Upload a CSV and ask natural language questions (e.g., "What is the average of column X?").
  - Groq API generates executable Python code to answer queries, with support for visualizations.

- **User-Friendly Interface**:
  - Built with Streamlit for an intuitive, responsive UI.
  - Custom styling with Roboto fonts and downloadable outputs.
  - Sidebar navigation for preprocessing, ML, and visualizations.
  - Example prompts for inspiration across domains like finance, healthcare, and education.

---

## Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Synthetic Data**: Faker
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn
- **Time Series**: Statsmodels (optional)
- **AI Integration**: Groq API
- **Deployment**: Hugging Face Spaces, Docker-compatible

---

## Getting Started

Follow these steps to set up and run DataGenie locally or deploy it to a cloud platform.

### Prerequisites

- **Python**: 3.8 or higher
- **Groq API Key**: Sign up at [xAI](https://x.ai/api) to obtain an API key.
- **Git**: For cloning the repository.

### Installation

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

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, install the following:
   ```bash
   pip install streamlit pandas numpy faker groq matplotlib seaborn plotly scikit-learn
   ```

   Optional (for time series):
   ```bash
   pip install statsmodels
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory:
     ```bash
     touch .env
     ```
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   - Open your browser to `http://localhost:8501` to access DataGenie.

---

## Deployment

### Hugging Face Spaces
DataGenie is deployed at [https://huggingface.co/spaces/MahatirTusher/DataGenie](https://huggingface.co/spaces/MahatirTusher/DataGenie). To deploy your own instance:

1. **Create a Hugging Face Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces).
   - Create a new Space with the Streamlit SDK.

2. **Upload Files**:
   - Push `app.py`, `requirements.txt`, and `.env` (with secrets configured in Space settings).

3. **Configure Secrets**:
   - Add your `GROQ_API_KEY` in the Space's secrets settings.

4. **Deploy**:
   - Commit changes to trigger a build. The Space will be live once the build completes.

### Local Deployment with Docker
1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app
   COPY . /app

   RUN pip install --no-cache-dir -r requirements.txt

   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Build and Run:
   ```bash
   docker build -t datagenie .
   docker run -p 8501:8501 --env-file .env datagenie
   ```

---

## Usage

1. **Generate Synthetic Data**:
   - Go to the "Dataset Generator" tab.
   - Enter a prompt (e.g., "Generate 1000 rows of patient data with age, blood pressure, and diagnosis").
   - Click "Generate Code" to preview the Python code, then "Get the Dataset" to view and download the data.

2. **Upload and Analyze Data**:
   - Upload a CSV in the header or "Chat with Dataset" tab.
   - Use the sidebar to clean data, detect outliers, encode variables, or visualize features.

3. **Train ML Models**:
   - Select "Select ML Models" in the sidebar.
   - Choose a task (classification, regression, clustering, time series), target/features, and model.
   - Train and view performance metrics.

4. **Chat with Dataset**:
   - Upload a CSV and ask questions like "Show the top 5 rows" or "Plot a histogram of column X".
   - Execute generated code to see results or visualizations.

5. **Explore Example Prompts**:
   - Check the "Example Prompts" tab for inspiration across domains like finance, healthcare, and retail.

---

## Example Prompts

Here are some prompts to try in the "Dataset Generator" tab:

- **Finance**: "Generate 1000 customer records for a bank with age, income, loan amount, credit score, and defaulted (Yes/No)."
- **Healthcare**: "Create 500 rows of patient data with age, gender, blood pressure, cholesterol, and diagnosis (Healthy, At Risk, Critical)."
- **Education**: "Generate 700 student records with study hours, attendance, and final grade (A, B, C, D, F)."
- **Retail**: "Generate 1000 rows of purchase data with customer ID, product category, purchase amount, and payment method."

See the "Example Prompts" tab in the app for more ideas!

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a pull request.

Please include tests and update documentation as needed.

---

## Issues and Support

If you encounter bugs or have feature requests:

- **Report Issues**: Use the [GitHub Issues](https://github.com/Mahatir-Ahmed-Tusher/DataGenie/issues) page.
- **Contact**: Reach out via [GitHub](https://github.com/Mahatir-Ahmed-Tusher).

---

## Acknowledgments

- **Developed by**: [Mahatir Ahmed Tusher](https://github.com/Mahatir-Ahmed-Tusher)
- **Inspired by**: Predicta by Ahmed Nafiz
- **Powered by**: [xAI's Groq API](https://x.ai), Streamlit, and open-source libraries

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Roadmap

Future enhancements planned for DataGenie:

- Advanced preprocessing (scaling, PCA, one-hot encoding).
- Model persistence (save/load trained models).
- Cloud storage integration (AWS S3, Google Drive).
- Interactive tutorials for new users.
- Unit tests for robust functionality.
- Support for additional ML frameworks (e.g., TensorFlow, PyTorch).

Stay tuned for updates!

---

*Empower your data journey with DataGenie!*


If you need help generating a `requirements.txt`, creating a `LICENSE` file, or adding specific sections (e.g., troubleshooting), let me know!
