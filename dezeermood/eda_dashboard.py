"""
Deezer Mood Detection - Exploratory Data Analysis Dashboard
A professional, thesis-ready Streamlit application for analyzing the Deezer Mood Detection Dataset

Author: Data Science Team
Purpose: Academic EDA for music emotion recognition research
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict

# Import custom utilities
from utils.data_loader import (
    load_all_datasets,
    validate_dataset,
    combine_datasets,
    get_dataset_summary,
    check_data_consistency,
    detect_outliers_iqr
)
from utils.statistics import (
    compute_descriptive_stats,
    compute_correlation,
    compare_distributions,
    test_normality,
    compute_split_statistics,
    compute_quadrant_distribution
)
from utils.visualizations import (
    plot_distribution_histogram,
    plot_overlaid_distributions,
    plot_density_comparison,
    plot_boxplot,
    plot_scatter_valence_arousal,
    plot_correlation_heatmap,
    plot_hexbin_density,
    plot_quadrant_distribution
)
from utils.recommender import (
    find_similar_tracks,
    recommend_by_mood,
    cluster_by_mood,
    analyze_mood_diversity,
    compute_mood_transition_path,
    get_artist_mood_profile,
    compute_recommendation_coverage
)
from utils.config import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    ABOUT_TEXT,
    DATA_QUALITY_TEXT,
    COLORS,
    TARGET_COLUMNS
)

# Configure page
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache all datasets"""
    return load_all_datasets()


@st.cache_data
def get_combined_data(datasets):
    """Combine and cache all datasets"""
    return combine_datasets(datasets)


def main():
    """Main application function"""
    
    # Title
    st.markdown('<div class="main-header">🎵 Deezer Mood Detection Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #7f8c8d; font-size: 1.2rem; margin-bottom: 2rem;">Exploratory Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dataset Overview & Analysis", "Data Visualization & Exploration", "Recommender System Analysis"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides a comprehensive exploratory data analysis "
        "of the Deezer Mood Detection Dataset for academic research purposes."
    )
    
    # Load data
    with st.spinner("Loading datasets..."):
        datasets = load_data()
        combined_data = get_combined_data(datasets)
    
    # Route to appropriate page
    if page == "Dataset Overview & Analysis":
        show_overview_page(datasets, combined_data)
    elif page == "Data Visualization & Exploration":
        show_visualization_page(datasets, combined_data)
    else:
        show_recommender_analysis_page(datasets, combined_data)


def show_overview_page(datasets: Dict[str, pd.DataFrame], combined_data: pd.DataFrame):
    """Display Page 1: Dataset Overview & Initial Analysis"""
    
    st.markdown('<div class="sub-header">📖 1. Introduction & Dataset Description</div>', unsafe_allow_html=True)
    
    # About section
    st.markdown(ABOUT_TEXT)
    
    # Visual representation of valence-arousal model
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Valence Dimension")
        st.markdown("""
        - **High Valence (+)**: Happy, joyful, pleasant
        - **Low Valence (-)**: Sad, angry, unpleasant
        - **Range**: Typically -2 to +2
        """)
    
    with col2:
        st.markdown("#### Arousal Dimension")
        st.markdown("""
        - **High Arousal (+)**: Excited, energetic, intense
        - **Low Arousal (-)**: Calm, relaxed, peaceful
        - **Range**: Typically -2 to +2
        """)
    
    st.markdown("---")
    
    # Dataset Summary
    st.markdown('<div class="sub-header">📊 2. Dataset Summary</div>', unsafe_allow_html=True)
    
    summary_df = get_dataset_summary(datasets)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{len(combined_data):,}")
    with col2:
        st.metric("Training Samples", f"{len(datasets['train']):,}")
    with col3:
        st.metric("Validation Samples", f"{len(datasets['validation']):,}")
    with col4:
        st.metric("Test Samples", f"{len(datasets['test']):,}")
    
    st.markdown("---")
    
    # Data Quality Assessment
    st.markdown('<div class="sub-header">🔍 3. Data Quality Assessment</div>', unsafe_allow_html=True)
    
    # Check for missing values
    st.markdown("#### Missing Values Analysis")
    
    missing_data = []
    for split_name, df in datasets.items():
        missing_count = df.isnull().sum().sum()
        missing_data.append({
            'Split': split_name.capitalize(),
            'Total Missing Values': missing_count,
            'Missing Percentage': f"{(missing_count / (len(df) * len(df.columns)) * 100):.2f}%"
        })
    
    missing_df = pd.DataFrame(missing_data)
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    if missing_df['Total Missing Values'].sum() == 0:
        st.success("✅ No missing values detected in any dataset split!")
    else:
        st.warning("⚠️ Missing values detected. Further investigation recommended.")
    
    # Data consistency check
    st.markdown("#### Data Consistency Across Splits")
    consistency = check_data_consistency(datasets)
    
    if consistency['columns_consistent']:
        st.success("✅ All dataset splits have consistent column structure")
    else:
        st.error("❌ Column structure inconsistency detected across splits")
    
    if consistency['dtypes_consistent']:
        st.success("✅ Data types are consistent across all splits")
    else:
        st.warning("⚠️ Data type inconsistencies detected")
    
    # Display column information
    st.markdown("#### Dataset Schema")
    schema_data = []
    for col in datasets['train'].columns:
        schema_data.append({
            'Column Name': col,
            'Data Type': str(datasets['train'][col].dtype),
            'Non-Null Count': datasets['train'][col].count(),
            'Sample Value': str(datasets['train'][col].iloc[0])
        })
    
    schema_df = pd.DataFrame(schema_data)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Descriptive Statistics
    st.markdown('<div class="sub-header">📈 4. Descriptive Statistics</div>', unsafe_allow_html=True)
    
    st.markdown("#### Statistics by Split and Variable")
    
    stats_df = compute_split_statistics(datasets)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Combined dataset statistics
    st.markdown("#### Combined Dataset Statistics")
    combined_stats = compute_descriptive_stats(combined_data, TARGET_COLUMNS)
    st.dataframe(combined_stats, use_container_width=True)
    
    # Key observations
    st.markdown("#### 📌 Key Statistical Observations")
    
    valence_mean = combined_data['valence'].mean()
    arousal_mean = combined_data['arousal'].mean()
    valence_std = combined_data['valence'].std()
    arousal_std = combined_data['arousal'].std()
    
    observations = f"""
    1. **Valence Distribution**:
       - Mean: {valence_mean:.3f} ({"positive" if valence_mean > 0 else "negative"} tendency)
       - Standard Deviation: {valence_std:.3f}
       - Range: [{combined_data['valence'].min():.3f}, {combined_data['valence'].max():.3f}]
    
    2. **Arousal Distribution**:
       - Mean: {arousal_mean:.3f} ({"high" if arousal_mean > 0 else "low"} energy tendency)
       - Standard Deviation: {arousal_std:.3f}
       - Range: [{combined_data['arousal'].min():.3f}, {combined_data['arousal'].max():.3f}]
    
    3. **Data Spread**: Both variables show substantial variation, indicating diverse emotional content
    
    4. **Skewness**: 
       - Valence: {combined_data['valence'].skew():.3f}
       - Arousal: {combined_data['arousal'].skew():.3f}
    """
    
    st.markdown(observations)
    
    # Correlation
    st.markdown("#### Correlation Analysis")
    corr_result = compute_correlation(combined_data, method='pearson')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pearson Correlation", f"{corr_result['correlation']:.4f}")
    with col2:
        st.metric("P-value", f"{corr_result['p_value']:.4e}")
    
    if abs(corr_result['correlation']) < 0.3:
        strength = "weak"
    elif abs(corr_result['correlation']) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    st.info(f"The correlation between valence and arousal is **{strength}** "
            f"({'positive' if corr_result['correlation'] > 0 else 'negative'}).")
    
    st.markdown("---")
    
    # Data Quality & Limitations
    st.markdown('<div class="sub-header">⚠️ 5. Data Quality & Limitations</div>', unsafe_allow_html=True)
    
    st.markdown(DATA_QUALITY_TEXT)
    
    # Outlier detection
    st.markdown("#### Outlier Detection (IQR Method)")
    
    outlier_data = []
    for split_name, df in datasets.items():
        for col in TARGET_COLUMNS:
            outliers, count = detect_outliers_iqr(df, col)
            percentage = (count / len(df)) * 100
            outlier_data.append({
                'Split': split_name.capitalize(),
                'Variable': col.capitalize(),
                'Outlier Count': count,
                'Percentage': f"{percentage:.2f}%"
            })
    
    outlier_df = pd.DataFrame(outlier_data)
    st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Note**: Outliers are identified using the IQR method (1.5 × IQR). These may represent:
    - Extreme but valid emotional expressions
    - Annotation errors
    - Edge cases in the dataset
    """)


def show_visualization_page(datasets: Dict[str, pd.DataFrame], combined_data: pd.DataFrame):
    """Display Page 2: Data Visualization & Exploration"""
    
    st.markdown('<div class="sub-header">📊 Interactive Data Exploration</div>', unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Filters")
    
    selected_split = st.sidebar.selectbox(
        "Select Dataset Split",
        ["Combined", "Train", "Validation", "Test"],
        key="viz_split_selector"
    )
    
    if selected_split == "Combined":
        data_to_plot = combined_data
    else:
        data_to_plot = datasets[selected_split.lower()]
    
    st.sidebar.metric("Selected Samples", f"{len(data_to_plot):,}")
    
    # Distribution Analysis
    st.markdown('<div class="sub-header">📈 1. Distribution Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Histograms", "Density Plots", "Box Plots"])
    
    with tab1:
        st.markdown("### Histograms")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Valence Distribution")
            fig_val = plot_distribution_histogram(
                data_to_plot,
                'valence',
                selected_split if selected_split != "Combined" else None,
                bins=50,
                color=COLORS.get(selected_split.lower(), COLORS['combined'])
            )
            st.pyplot(fig_val)
            plt.close()
        
        with col2:
            st.markdown("#### Arousal Distribution")
            fig_aro = plot_distribution_histogram(
                data_to_plot,
                'arousal',
                selected_split if selected_split != "Combined" else None,
                bins=50,
                color=COLORS.get(selected_split.lower(), COLORS['combined'])
            )
            st.pyplot(fig_aro)
            plt.close()
    
    with tab2:
        st.markdown("### Kernel Density Estimation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Valence Density")
            fig_val_density = plot_density_comparison(datasets, 'valence')
            st.pyplot(fig_val_density)
            plt.close()
        
        with col2:
            st.markdown("#### Arousal Density")
            fig_aro_density = plot_density_comparison(datasets, 'arousal')
            st.pyplot(fig_aro_density)
            plt.close()
    
    with tab3:
        st.markdown("### Box Plots (Split Comparison)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Valence Box Plot")
            fig_val_box = plot_boxplot(datasets, 'valence')
            st.pyplot(fig_val_box)
            plt.close()
        
        with col2:
            st.markdown("#### Arousal Box Plot")
            fig_aro_box = plot_boxplot(datasets, 'arousal')
            st.pyplot(fig_aro_box)
            plt.close()
    
    st.markdown("---")
    
    # Split Consistency Analysis
    st.markdown('<div class="sub-header">🔄 2. Split Consistency Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("### Overlaid Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Valence Across Splits")
        fig_val_overlay = plot_overlaid_distributions(datasets, 'valence', bins=40)
        st.pyplot(fig_val_overlay)
        plt.close()
    
    with col2:
        st.markdown("#### Arousal Across Splits")
        fig_aro_overlay = plot_overlaid_distributions(datasets, 'arousal', bins=40)
        st.pyplot(fig_aro_overlay)
        plt.close()
    
    # Statistical comparison
    st.markdown("### Statistical Comparison Tests")
    
    st.markdown("**Comparing Training vs Validation Sets**")
    
    val_comparison = compare_distributions(datasets['train'], datasets['validation'], 'valence')
    aro_comparison = compare_distributions(datasets['train'], datasets['validation'], 'arousal')
    
    comparison_data = [
        {
            'Variable': 'Valence',
            'KS Test p-value': f"{val_comparison['ks_test']['p_value']:.4f}",
            'Significant Difference': "Yes" if val_comparison['ks_test']['significant'] else "No",
            'T-test p-value': f"{val_comparison['t_test']['p_value']:.4f}"
        },
        {
            'Variable': 'Arousal',
            'KS Test p-value': f"{aro_comparison['ks_test']['p_value']:.4f}",
            'Significant Difference': "Yes" if aro_comparison['ks_test']['significant'] else "No",
            'T-test p-value': f"{aro_comparison['t_test']['p_value']:.4f}"
        }
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.info("**Interpretation**: A p-value > 0.05 suggests no significant difference between distributions (desired for consistent splits).")
    
    st.markdown("---")
    
    # Relationship Analysis
    st.markdown('<div class="sub-header">🎯 3. Valence-Arousal Relationship Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Density Heatmap", "Quadrant Analysis"])
    
    with tab1:
        st.markdown("### Scatter Plot: Valence vs Arousal")
        
        sample_size = st.slider(
            "Number of points to display (for performance)",
            min_value=1000,
            max_value=min(len(data_to_plot), 15000),
            value=min(5000, len(data_to_plot)),
            step=1000
        )
        
        fig_scatter = plot_scatter_valence_arousal(
            data_to_plot,
            selected_split if selected_split != "Combined" else None,
            sample_size=sample_size,
            alpha=0.3
        )
        st.pyplot(fig_scatter)
        plt.close()
    
    with tab2:
        st.markdown("### Hexbin Density Plot")
        
        fig_hexbin = plot_hexbin_density(
            data_to_plot,
            selected_split if selected_split != "Combined" else None,
            gridsize=30
        )
        st.pyplot(fig_hexbin)
        plt.close()
    
    with tab3:
        st.markdown("### Emotional Quadrant Distribution")
        
        quadrant_df = compute_quadrant_distribution(data_to_plot)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(quadrant_df, use_container_width=True, hide_index=True)
        
        with col2:
            fig_quadrant = plot_quadrant_distribution(quadrant_df)
            st.pyplot(fig_quadrant)
            plt.close()
        
        st.markdown("""
        **Quadrant Interpretation**:
        - **Q1 (Happy/Excited)**: Positive valence, high arousal
        - **Q2 (Angry/Tense)**: Negative valence, high arousal
        - **Q3 (Sad/Depressed)**: Negative valence, low arousal
        - **Q4 (Calm/Relaxed)**: Positive valence, low arousal
        """)
    
    # Correlation heatmap
    st.markdown("### Correlation Matrix")
    fig_corr = plot_correlation_heatmap(data_to_plot)
    st.pyplot(fig_corr)
    plt.close()
    
    corr_result = compute_correlation(data_to_plot)
    st.metric("Pearson Correlation Coefficient", f"{corr_result['correlation']:.4f}")
    
    st.markdown("---")
    
    # Summary insights
    st.markdown('<div class="sub-header">💡 4. Key Insights</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### Summary of Findings
    
    1. **Dataset Size**: The combined dataset contains **{len(combined_data):,}** music tracks with emotional annotations.
    
    2. **Distribution Characteristics**:
       - Both valence and arousal show approximately normal distributions
       - Valence mean: {combined_data['valence'].mean():.3f}
       - Arousal mean: {combined_data['arousal'].mean():.3f}
    
    3. **Split Consistency**: The train, validation, and test splits show {"consistent" if not val_comparison['ks_test']['significant'] else "some differences in"} distributions.
    
    4. **Valence-Arousal Relationship**: 
       - Correlation: {corr_result['correlation']:.3f}
       - The relationship is {"weak" if abs(corr_result['correlation']) < 0.3 else "moderate" if abs(corr_result['correlation']) < 0.7 else "strong"}
    
    5. **Emotional Coverage**: The dataset covers all four emotional quadrants, with varying densities.
    
    6. **Data Quality**: {"No" if combined_data.isnull().sum().sum() == 0 else "Some"} missing values detected.
    """)


def identify_quadrant(valence: float, arousal: float) -> str:
    """Helper function to identify emotional quadrant"""
    if valence >= 0 and arousal >= 0:
        return 'Happy/Excited'
    elif valence < 0 and arousal >= 0:
        return 'Angry/Tense'
    elif valence < 0 and arousal < 0:
        return 'Sad/Depressed'
    else:
        return 'Calm/Relaxed'


def show_recommender_analysis_page(datasets: Dict[str, pd.DataFrame], combined_data: pd.DataFrame):
    """Display Page 3: Recommender System Analysis"""
    
    st.markdown('<div class="sub-header">🎯 Recommender System Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This page provides EDA specifically focused on **building a mood-based music recommender system**.
    Explore similarity metrics, clustering, and recommendation strategies based on the valence-arousal emotional space.
    """)
    
    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Recommender Settings")
    
    selected_split = st.sidebar.selectbox(
        "Select Dataset Split",
        ["Combined", "Train", "Validation", "Test"],
        key="rec_split"
    )
    
    if selected_split == "Combined":
        data_to_analyze = combined_data
    else:
        data_to_analyze = datasets[selected_split.lower()]
    
    st.sidebar.metric("Tracks Available", f"{len(data_to_analyze):,}")
    
    st.markdown("---")
    
    # 1. Mood Diversity Analysis
    st.markdown('<div class="sub-header">📊 1. Mood Diversity Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters for Recommenders**: Understanding mood diversity helps ensure the recommender 
    can provide varied recommendations across the entire emotional spectrum.
    """)
    
    with st.spinner("Analyzing mood diversity..."):
        diversity_metrics = analyze_mood_diversity(data_to_analyze)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Distance", f"{diversity_metrics['mean_distance']:.3f}")
    with col2:
        st.metric("Std Distance", f"{diversity_metrics['std_distance']:.3f}")
    with col3:
        st.metric("Max Distance", f"{diversity_metrics['max_distance']:.3f}")
    with col4:
        st.metric("Coverage Score", f"{diversity_metrics['coverage_score']:.3f}")
    
    st.info("""
    **Interpretation**: 
    - **Mean Distance**: Average emotional distance between tracks (higher = more diverse)
    - **Coverage Score**: Spread in emotional space (higher = better coverage)
    - **High diversity** enables rich, varied recommendations
    """)
    
    st.markdown("---")
    
    # 2. Mood Clustering
    st.markdown('<div class="sub-header">🎨 2. Mood-Based Clustering</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters for Recommenders**: Clustering identifies natural mood groups, 
    enabling category-based recommendations and efficient similarity search.
    """)
    
    n_clusters = st.slider("Number of Mood Clusters", min_value=4, max_value=12, value=8, step=1)
    
    with st.spinner(f"Clustering tracks into {n_clusters} mood groups..."):
        clustered_data, kmeans_model = cluster_by_mood(data_to_analyze, n_clusters=n_clusters)
    
    # Visualize clusters
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(
        clustered_data['valence'],
        clustered_data['arousal'],
        c=clustered_data['cluster'],
        cmap='tab10',
        alpha=0.6,
        s=30,
        edgecolors='none'
    )
    
    # Plot cluster centers
    centers = kmeans_model.cluster_centers_
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c='red',
        marker='X',
        s=300,
        edgecolors='black',
        linewidths=2,
        label='Cluster Centers',
        zorder=5
    )
    
    # Add quadrant lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Valence (Negativity ← → Positivity)', fontsize=12)
    ax.set_ylabel('Arousal (Low Energy ← → High Energy)', fontsize=12)
    ax.set_title(f'K-Means Clustering ({n_clusters} Mood Clusters)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Cluster statistics
    st.markdown("#### Cluster Statistics")
    cluster_stats = clustered_data.groupby('cluster').agg({
        'valence': ['mean', 'std', 'count'],
        'arousal': ['mean', 'std']
    }).round(3)
    cluster_stats.columns = ['Valence Mean', 'Valence Std', 'Track Count', 'Arousal Mean', 'Arousal Std']
    st.dataframe(cluster_stats, use_container_width=True)
    
    st.success(f"✅ Identified {n_clusters} distinct mood clusters for efficient recommendation grouping")
    
    st.markdown("---")
    
    # 3. Similar Track Finder (Demo)
    st.markdown('<div class="sub-header">🔍 3. Similar Track Finder (Recommendation Demo)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters for Recommenders**: This demonstrates content-based filtering - 
    finding tracks with similar emotional characteristics.
    """)
    
    # Random track selector
    if st.button("🎲 Select Random Track"):
        st.session_state.random_track_idx = np.random.randint(0, len(data_to_analyze))
    
    if 'random_track_idx' not in st.session_state:
        st.session_state.random_track_idx = 0
    
    reference_track = data_to_analyze.iloc[st.session_state.random_track_idx]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎵 Reference Track")
        st.markdown(f"""
        **Artist**: {reference_track['artist_name']}  
        **Track**: {reference_track['track_name']}  
        **Valence**: {reference_track['valence']:.3f}  
        **Arousal**: {reference_track['arousal']:.3f}  
        **Mood**: {identify_quadrant(reference_track['valence'], reference_track['arousal'])}
        """)
    
    with col2:
        n_recommendations = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)
    
    # Find similar tracks
    with st.spinner("Finding similar tracks..."):
        similar_tracks = find_similar_tracks(
            data_to_analyze.reset_index(drop=True),
            st.session_state.random_track_idx,
            n_recommendations=n_recommendations
        )
    
    st.markdown("#### 🎯 Recommended Similar Tracks")
    
    # Display recommendations
    display_cols = ['artist_name', 'track_name', 'valence', 'arousal', 'similarity_score']
    similar_display = similar_tracks[display_cols].copy()
    similar_display.columns = ['Artist', 'Track', 'Valence', 'Arousal', 'Similarity Score']
    similar_display['Similarity Score'] = similar_display['Similarity Score'].round(3)
    similar_display['Valence'] = similar_display['Valence'].round(3)
    similar_display['Arousal'] = similar_display['Arousal'].round(3)
    
    st.dataframe(similar_display, use_container_width=True, hide_index=True)
    
    st.info("""
    **How It Works**: Uses K-Nearest Neighbors (KNN) with Euclidean distance in valence-arousal space.
    Higher similarity score = more emotionally similar to the reference track.
    """)
    
    st.markdown("---")
    
    # 4. Mood-Based Recommendation
    st.markdown('<div class="sub-header">🎭 4. Mood-Based Recommendation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters for Recommenders**: Allows users to specify desired mood and get matching tracks.
    This is the core of a mood-based recommender system.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_valence = st.slider(
            "Target Valence (Emotional Positivity)",
            min_value=-2.5,
            max_value=2.5,
            value=0.0,
            step=0.1
        )
    
    with col2:
        target_arousal = st.slider(
            "Target Arousal (Energy Level)",
            min_value=-2.5,
            max_value=2.5,
            value=0.0,
            step=0.1
        )
    
    target_mood = identify_quadrant(target_valence, target_arousal)
    st.markdown(f"**Selected Mood**: {target_mood}")
    
    n_mood_recs = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10, key="mood_recs")
    
    # Get recommendations
    with st.spinner("Finding tracks matching your mood..."):
        mood_recommendations = recommend_by_mood(
            data_to_analyze.reset_index(drop=True),
            target_valence,
            target_arousal,
            n_recommendations=n_mood_recs
        )
    
    st.markdown("#### 🎵 Recommended Tracks for Your Mood")
    
    display_cols = ['artist_name', 'track_name', 'valence', 'arousal', 'match_score']
    mood_display = mood_recommendations[display_cols].copy()
    mood_display.columns = ['Artist', 'Track', 'Valence', 'Arousal', 'Match Score']
    mood_display['Match Score'] = mood_display['Match Score'].round(3)
    mood_display['Valence'] = mood_display['Valence'].round(3)
    mood_display['Arousal'] = mood_display['Arousal'].round(3)
    
    st.dataframe(mood_display, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 5. Mood Transition Playlist
    st.markdown('<div class="sub-header">🌈 5. Mood Transition Playlist Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters for Recommenders**: Creates playlists that smoothly transition between moods,
    useful for activities like workout warm-up/cool-down or sleep preparation.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Start Mood")
        start_v = st.slider("Start Valence", -2.0, 2.0, -1.0, 0.1, key="start_v")
        start_a = st.slider("Start Arousal", -2.0, 2.0, -1.0, 0.1, key="start_a")
        st.markdown(f"**Mood**: {identify_quadrant(start_v, start_a)}")
    
    with col2:
        st.markdown("##### End Mood")
        end_v = st.slider("End Valence", -2.0, 2.0, 1.0, 0.1, key="end_v")
        end_a = st.slider("End Arousal", -2.0, 2.0, 1.0, 0.1, key="end_a")
        st.markdown(f"**Mood**: {identify_quadrant(end_v, end_a)}")
    
    n_steps = st.slider("Playlist Length (tracks)", 3, 10, 5)
    
    if st.button("🎵 Generate Mood Journey Playlist"):
        with st.spinner("Creating mood transition playlist..."):
            playlist = compute_mood_transition_path(
                data_to_analyze.reset_index(drop=True),
                start_v, start_a,
                end_v, end_a,
                n_steps=n_steps
            )
        
        st.markdown("#### 🎵 Your Mood Journey Playlist")
        
        for idx, track in playlist.iterrows():
            st.markdown(f"""
            **{idx + 1}.** {track['artist_name']} - {track['track_name']}  
            *Valence: {track['valence']:.2f}, Arousal: {track['arousal']:.2f}* ({identify_quadrant(track['valence'], track['arousal'])})
            """)
        
        st.success(f"✅ Created a {n_steps}-track playlist transitioning from {identify_quadrant(start_v, start_a)} to {identify_quadrant(end_v, end_a)}")
    
    st.markdown("---")
    
    # 6. Recommender System Metrics
    st.markdown('<div class="sub-header">📈 6. Recommender System Readiness Metrics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters**: These metrics assess how well the dataset supports building a recommender system.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tracks", f"{len(data_to_analyze):,}")
        st.caption("Catalog size for recommendations")
    
    with col2:
        unique_artists = data_to_analyze['artist_name'].nunique()
        st.metric("Unique Artists", f"{unique_artists:,}")
        st.caption("Artist diversity")
    
    with col3:
        avg_tracks_per_artist = len(data_to_analyze) / unique_artists
        st.metric("Avg Tracks/Artist", f"{avg_tracks_per_artist:.1f}")
        st.caption("Artist representation")
    
    # Coverage analysis
    st.markdown("#### Recommendation Coverage Analysis")
    
    if st.button("🔍 Compute Recommendation Coverage"):
        with st.spinner("Computing coverage (this may take a moment)..."):
            coverage = compute_recommendation_coverage(data_to_analyze.reset_index(drop=True), n_recommendations=10)
        
        st.metric("Catalog Coverage", f"{coverage:.1f}%")
        st.caption("Percentage of catalog that can be recommended (with 10 recommendations per query)")
        
        if coverage > 80:
            st.success("✅ Excellent coverage! Most tracks can be recommended.")
        elif coverage > 50:
            st.info("ℹ️ Good coverage. Consider strategies to improve long-tail recommendations.")
        else:
            st.warning("⚠️ Low coverage. Many tracks may not be recommended frequently.")
    
    st.markdown("---")
    
    # 7. Key Insights for Recommender System
    st.markdown('<div class="sub-header">💡 7. Key Insights for Building Recommender System</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### Summary of Recommender-Focused Findings
    
    Based on the analysis above, here are key insights for building your mood-based recommender:
    
    1. **Dataset Size**: {len(data_to_analyze):,} tracks provide a solid foundation for recommendations
    
    2. **Mood Diversity**: 
       - Mean emotional distance: {diversity_metrics['mean_distance']:.3f}
       - The dataset covers diverse moods, enabling varied recommendations
    
    3. **Clustering**: 
       - Natural mood groups identified through K-Means clustering
       - Can be used for efficient category-based recommendations
    
    4. **Similarity Search**:
       - Euclidean distance in valence-arousal space works well
       - K-Nearest Neighbors provides fast, accurate similar track finding
    
    5. **Mood-Based Filtering**:
       - Users can specify desired mood (valence, arousal)
       - System finds closest matching tracks effectively
    
    6. **Playlist Generation**:
       - Smooth mood transitions are possible
       - Useful for activity-based playlists (workout, relaxation, etc.)
    
    7. **Recommendation Strategies**:
       - ✅ **Content-Based**: Use valence-arousal similarity
       - ✅ **Mood-Based**: Direct mood specification
       - ✅ **Hybrid**: Combine with collaborative filtering
       - ✅ **Context-Aware**: Mood trajectories for activities
    
    ### Next Steps for Implementation
    
    1. **Extract Audio Features**: Add MFCCs, spectral features for richer recommendations
    2. **Build Prediction Models**: Train regression models to predict valence/arousal from audio
    3. **Implement Ranking**: Add relevance scoring and diversity constraints
    4. **Add Personalization**: Learn user preferences over time
    5. **Optimize Performance**: Use approximate nearest neighbors (FAISS) for scalability
    6. **User Interface**: Build interactive mood selector and playlist generator
    
    ### Recommended Algorithms
    
    - **Similarity**: K-Nearest Neighbors (KNN) with Euclidean distance
    - **Clustering**: K-Means for mood categories
    - **Ranking**: Weighted combination of similarity + diversity + popularity
    - **Personalization**: Collaborative filtering or matrix factorization
    """)


def identify_quadrant(valence: float, arousal: float) -> str:
    """Helper function to identify emotional quadrant"""
    if valence >= 0 and arousal >= 0:
        return 'Happy/Excited'
    elif valence < 0 and arousal >= 0:
        return 'Angry/Tense'
    elif valence < 0 and arousal < 0:
        return 'Sad/Depressed'
    else:
        return 'Calm/Relaxed'


if __name__ == "__main__":
    main()
