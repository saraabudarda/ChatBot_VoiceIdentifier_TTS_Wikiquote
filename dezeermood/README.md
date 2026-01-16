# Deezer Mood Detection - EDA Dashboard

A professional, thesis-ready Streamlit dashboard for comprehensive Exploratory Data Analysis (EDA) of the Deezer Mood Detection Dataset.

## 📋 Overview

This dashboard provides an academically rigorous analysis of music emotion recognition data, focusing on the valence-arousal emotional model. It is designed for research purposes and thesis submissions.

## 🎯 Features

### Page 1: Dataset Overview & Initial Analysis
- **Introduction**: Comprehensive explanation of the valence-arousal model
- **Dataset Summary**: Sample counts, split information, and feature overview
- **Data Quality Assessment**: Missing values, consistency checks, schema validation
- **Descriptive Statistics**: Comprehensive statistics for all splits
- **Correlation Analysis**: Relationship between valence and arousal
- **Outlier Detection**: IQR-based outlier identification

### Page 2: Data Visualization & Exploration
- **Distribution Analysis**: Histograms, density plots, and box plots
- **Split Consistency**: Statistical comparison across train/validation/test
- **Relationship Analysis**: Scatter plots, hexbin density, correlation heatmaps
- **Quadrant Analysis**: Emotional category distribution
- **Interactive Filters**: Dataset split selection and dynamic visualizations

## 📁 Project Structure

```
dezeermood/
├── train.csv                  # Training dataset
├── validation.csv             # Validation dataset
├── test.csv                   # Test dataset
├── eda_dashboard.py           # Main Streamlit application
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── config.py             # Configuration and constants
│   ├── data_loader.py        # Data loading and validation
│   ├── statistics.py         # Statistical analysis functions
│   └── visualizations.py     # Plotting functions
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Navigate to the project directory**:
   ```bash
   cd "/Users/sara/Desktop/wiki db/dezeermood"
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Dashboard

```bash
streamlit run eda_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Navigation

1. **Sidebar**: Use the navigation menu to switch between pages
2. **Filters**: Select dataset splits and adjust visualization parameters
3. **Interactive Elements**: Hover over plots for detailed information

## 📊 Dataset Information

### Files
- **train.csv**: 11,269 samples for model training
- **validation.csv**: ~4,000 samples for hyperparameter tuning
- **test.csv**: 3,516 samples for final evaluation

### Features
- `dzr_sng_id`: Deezer song identifier
- `MSD_sng_id`: Million Song Dataset song ID
- `MSD_track_id`: Million Song Dataset track ID
- `valence`: Emotional positivity (-2 to +2)
- `arousal`: Emotional intensity (-2 to +2)
- `artist_name`: Artist name
- `track_name`: Track title

### Valence-Arousal Model

The dataset uses Russell's Circumplex Model of Affect:

- **Valence**: Represents emotional positivity/negativity
  - Positive: Happy, joyful, pleasant
  - Negative: Sad, angry, unpleasant

- **Arousal**: Represents emotional intensity/energy
  - High: Excited, energetic, intense
  - Low: Calm, relaxed, peaceful

### Emotional Quadrants
- **Q1 (Happy/Excited)**: Positive valence, high arousal
- **Q2 (Angry/Tense)**: Negative valence, high arousal
- **Q3 (Sad/Depressed)**: Negative valence, low arousal
- **Q4 (Calm/Relaxed)**: Positive valence, low arousal

## 🔬 Analysis Components

### Statistical Tests
- **Descriptive Statistics**: Mean, std, min, max, quartiles, skewness, kurtosis
- **Correlation Analysis**: Pearson and Spearman coefficients
- **Distribution Comparison**: Kolmogorov-Smirnov, Mann-Whitney U, t-tests
- **Normality Tests**: Shapiro-Wilk, Anderson-Darling, KS tests
- **Outlier Detection**: IQR method (1.5 × IQR)

### Visualizations
- **Histograms**: Distribution of valence and arousal
- **Density Plots**: Kernel density estimation
- **Box Plots**: Outlier visualization and split comparison
- **Scatter Plots**: Valence-arousal relationship
- **Hexbin Plots**: High-density visualization
- **Correlation Heatmaps**: Variable relationships
- **Quadrant Plots**: Emotional category distribution

## 📝 Academic Use

This dashboard is designed for:
- Thesis submissions
- Research presentations
- Academic papers
- Data science coursework

### Citation-Ready Features
- Professional visualizations
- Comprehensive statistical analysis
- Clear explanations and interpretations
- Reproducible methodology

## 🛠️ Technical Details

### Dependencies
- **streamlit**: Web dashboard framework
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**: Base plotting
- **seaborn**: Statistical visualizations
- **scipy**: Statistical tests

### Code Quality
- Modular architecture
- Type hints
- Comprehensive docstrings
- Error handling
- Consistent styling

## 📌 Key Insights

The dashboard automatically computes and displays:
1. Dataset size and split distribution
2. Distribution characteristics (normality, skewness, kurtosis)
3. Split consistency (statistical tests)
4. Valence-arousal correlation
5. Emotional quadrant coverage
6. Data quality metrics
7. Outlier analysis

## 🤝 Contributing

This is an academic project. For improvements or suggestions:
1. Document the proposed change
2. Ensure academic rigor
3. Maintain code quality standards
4. Test thoroughly

## 📄 License

This project is for academic and research purposes.

## 👥 Author

Data Science Team - Academic Research Project

## 📧 Support

For questions or issues, please refer to the project documentation or contact your academic supervisor.

---

**Note**: This dashboard is designed for exploratory data analysis and does not include machine learning model training or prediction functionality. It focuses on understanding the dataset characteristics and preparing for subsequent modeling work.
