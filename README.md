# Employee Attrition Prediction

A machine learning-powered web application that predicts employee attrition using advanced ensemble methods. Built with **AdaBoost** classifier and deployed as an interactive **Streamlit** dashboard.

![Employee Attrition Dashboard](https://via.placeholder.com/800x400.png?text=Employee+Attrition+Dashboard+Screenshot)

## Problem Statement

Employee attrition is one of the most significant challenges in HR analytics. Replacing an employee costs nearly **50–250% of their annual salary**. This predictive model helps organizations:

- Identify employees at risk of leaving early
- Enable proactive retention strategies
- Reduce recruitment and training costs
- Improve workforce planning and stability

## Features

- **Real-time Prediction**: Instant attrition risk assessment
- **Comprehensive Input Form**: 24 employee attributes across personal, educational, and professional domains
- **High Accuracy**: 93.8% F1-score on test data with low overfitting risk
- **Interactive UI**: Clean, intuitive Streamlit interface with visual feedback
- **Production-Ready**: Complete preprocessing pipeline saved as a single artifact

## Project Structure

```
employee_attrition_prediction/
├── app.py                              # Streamlit web application
├── employee_attrition_prediction.ipynb # Complete ML pipeline (EDA, training, evaluation)
├── employee_attrition_model.joblib     # Trained model + preprocessing artifacts
├── data.csv                            # Dataset (59,603 records, 24 features)
├── requirements.txt                    # Python dependencies
├── .github/
│   └── copilot-instructions.md         # AI agent development guide
└── README.md                           # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BharathVasanth12/employee-attrition-prediction.git
   cd employee-attrition-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud

### Option 1: Deploy via Streamlit Cloud UI

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Click "New app"** and connect your GitHub repository

4. **Configure deployment settings:**
   - Repository: `BharathVasanth12/employee-attrition-prediction`
   - Branch: `main`
   - Main file path: `app.py`
   - Python version: `3.9` (or higher)

5. **Click "Deploy"** and wait for the app to build

### Option 2: Deploy via `.streamlit/config.toml`

Create a `.streamlit` folder with configuration:

```bash
mkdir -p .streamlit
```

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor="#667eea"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f0f2f6"
textColor="#262730"
font="sans serif"

[server]
headless = true
port = 8501
enableCORS = false
```

### Option 3: Deploy via Runtime Configuration

Create a `.streamlit/secrets.toml` file for any API keys or secrets (optional):
```toml
# Add your secrets here
# api_key = "your-secret-key"
```

Then follow the same steps as Option 1.

## Dataset

- **Source**: [Kaggle - Employee Attrition Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset)
- **Size**: 59,603 employee records
- **Features**: 24 attributes including:
  - **Personal**: Age, Gender, Marital Status, Number of Dependents
  - **Educational**: Education Level, Job Role
  - **Professional**: Years at Company, Job Satisfaction, Performance Rating, Monthly Income, Overtime, Work-Life Balance
  - **Workplace**: Company Size, Remote Work, Leadership Opportunities, Employee Recognition
- **Target**: Binary classification (Stayed / Left)

## Model Architecture

### Preprocessing Pipeline

1. **Missing Value Imputation**
   - Numeric: Median (if outliers exist) or Mean
   - Categorical: Mode

2. **Encoding Strategy** (order matters!)
   - **Binary Encoding**: Gender, Overtime, Remote Work, Leadership/Innovation Opportunities (6 columns)
   - **Ordinal Encoding**: Job Satisfaction, Education Level, Job Level, etc. (8 columns)
   - **OneHot Encoding**: Job Role, Marital Status (drop='first')

3. **Feature Engineering**
   - Combined `Leadership Opportunities` + `Innovation Opportunities` → `Opportunities`

4. **Outlier Handling**
   - `Years at Company`: Capped at 40 years (max observed: 51)

5. **Skewness Correction**
   - PowerTransformer (Yeo-Johnson) applied to: `Number of Dependents`, `Number of Promotions`, `Years at Company`

6. **Scaling**
   - StandardScaler fit on training data

### Model Performance

| Metric | Training | Testing | 10-Fold CV |
|--------|----------|---------|------------|
| Accuracy | 96.5% | 94.1% | 94.0% |
| Precision | 96.4% | 93.9% | - |
| Recall | 96.4% | 93.9% | - |
| **F1-Score** | **96.3%** | **93.8%** | **93.7%** |
| **Overfitting Risk** | - | - | **Low** |

**Algorithm**: AdaBoostClassifier (random_state=42, default hyperparameters)

## Usage

### Web Application

1. Navigate to the deployed app or run locally
2. Fill in employee details across three sections:
   - 🧍 **Personal Details**: Age, Gender, Marital Status, etc.
   - 🎓 **Education & Career Level**: Education, Job Role, Job Level
   - 🏢 **Workplace & Performance**: Years at Company, Salary, Satisfaction, etc.
3. Click **"Predict Attrition"**
4. View instant prediction: ✅ Employee Will Stay or 🚨 Employee Will Leave

### Retraining the Model

Execute the Jupyter notebook cells sequentially:

```bash
jupyter notebook employee_attrition_prediction.ipynb
```

**Critical cell sequence:**
1. Load data + EDA (Cells 1-29)
2. Binary → Ordinal → OneHot encoding (Cells 30-40)
3. Outlier capping + Skewness transformation (Cells 41-45)
4. Feature engineering (Cell 46)
5. Train/test split + Scaling (Cells 47-50)
6. AdaBoost training (Cells 96-98)
7. Save artifact (Cell 99) → generates `employee_attrition_model.joblib`

## 📦 Dependencies

Core libraries:
```
streamlit
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
scipy
joblib
imbalanced-learn
```

Optional (for experiment tracking):
```
mlflow<3
dvc
dvc-s3
```

See `requirements.txt` for complete list.

## Configuration

### Model Artifact Structure

The `employee_attrition_model.joblib` contains:
- Trained AdaBoost model
- OneHotEncoder (for Job Role, Marital Status)
- StandardScaler
- PowerTransformer (for skewness correction)
- Binary encoding mappings
- Ordinal encoding mappings
- Feature column names (for alignment)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Dataset: [Stealth Technologies on Kaggle](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset)
- Built with [Streamlit](https://streamlit.io/)
- ML framework: [scikit-learn](https://scikit-learn.org/)

## Contact

For questions or feedback, please open an issue or reach out via GitHub.

---

**Star this repo** if you find it helpful!
