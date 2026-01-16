"""
Configuration file for the Deezer Mood Detection EDA Dashboard
Contains constants, color schemes, and settings
"""

# Dataset paths
DATA_DIR = "/Users/sara/Desktop/wiki db/dezeermood"
TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"

# Column names
ID_COLUMNS = ['dzr_sng_id', 'MSD_sng_id', 'MSD_track_id']
TARGET_COLUMNS = ['valence', 'arousal']
METADATA_COLUMNS = ['artist_name', 'track_name']
ALL_COLUMNS = ID_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS

# Valence-Arousal ranges (typical range based on dataset)
VALENCE_RANGE = (-2.5, 2.5)
AROUSAL_RANGE = (-2.5, 2.5)

# Color schemes for visualizations
COLORS = {
    'train': '#3498db',      # Blue
    'validation': '#e74c3c', # Red
    'test': '#2ecc71',       # Green
    'combined': '#9b59b6',   # Purple
    'primary': '#2c3e50',    # Dark blue-gray
    'secondary': '#95a5a6',  # Gray
}

# Plot styling
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (10, 6)
FONT_SIZE = 12
TITLE_SIZE = 14

# Streamlit page configuration
PAGE_TITLE = "Deezer Mood Detection - EDA Dashboard"
PAGE_ICON = "🎵"
LAYOUT = "wide"

# Emotional quadrants (based on valence-arousal model)
EMOTIONAL_QUADRANTS = {
    'Q1': {'name': 'Happy/Excited', 'valence': 'positive', 'arousal': 'high'},
    'Q2': {'name': 'Angry/Tense', 'valence': 'negative', 'arousal': 'high'},
    'Q3': {'name': 'Sad/Depressed', 'valence': 'negative', 'arousal': 'low'},
    'Q4': {'name': 'Calm/Relaxed', 'valence': 'positive', 'arousal': 'low'},
}

# Statistical significance level
ALPHA = 0.05

# Dashboard text content
ABOUT_TEXT = """
## About the Dataset

The **Deezer Mood Detection Dataset** is designed for music emotion recognition research. 
It contains metadata and emotional annotations for music tracks, focusing on two key dimensions 
of emotion: **valence** and **arousal**.

### Valence-Arousal Model (Russell's Circumplex Model)

The dataset uses the **valence-arousal emotional space**, a well-established psychological model:

- **Valence**: Represents the positivity or negativity of emotion
  - Positive values → Happy, joyful, pleasant emotions
  - Negative values → Sad, angry, unpleasant emotions

- **Arousal**: Represents the intensity or energy level of emotion
  - High values → Excited, energetic, intense emotions
  - Low values → Calm, relaxed, low-energy emotions

### Task Type

This is a **regression problem** where the goal is to predict continuous values for valence 
and arousal based on audio features (not included in this dataset).

### Dataset Splits

- **Training set**: Used to train machine learning models
- **Validation set**: Used to tune hyperparameters and prevent overfitting
- **Test set**: Used for final model evaluation
"""

DATA_QUALITY_TEXT = """
### Data Quality Considerations

1. **Label Subjectivity**: Emotional annotations are inherently subjective and may vary between annotators
2. **Missing Audio Features**: This dataset contains only metadata and labels, not raw audio or extracted features
3. **Outliers**: Some tracks may have extreme valence/arousal values that could represent annotation errors or genuinely extreme emotions
4. **Distribution Balance**: The distribution of emotions across the valence-arousal space may not be uniform
"""
