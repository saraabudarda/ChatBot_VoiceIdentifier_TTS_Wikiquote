"""
Recommender system utilities for Deezer Mood Detection Dataset
Functions to support building a mood-based music recommender
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import euclidean, cosine
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def compute_similarity_matrix(df: pd.DataFrame, method: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise similarity matrix for all tracks.
    
    Args:
        df: DataFrame with valence and arousal columns
        method: Similarity method ('euclidean' or 'cosine')
        
    Returns:
        Similarity matrix (n_samples x n_samples)
    """
    features = df[['valence', 'arousal']].values
    n_samples = len(features)
    similarity_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            if method == 'euclidean':
                similarity_matrix[i, j] = euclidean(features[i], features[j])
            elif method == 'cosine':
                similarity_matrix[i, j] = 1 - cosine(features[i], features[j])
    
    return similarity_matrix


def find_similar_tracks(
    df: pd.DataFrame,
    track_idx: int,
    n_recommendations: int = 10,
    method: str = 'euclidean'
) -> pd.DataFrame:
    """
    Find similar tracks based on emotional similarity.
    
    Args:
        df: DataFrame with track information
        track_idx: Index of the reference track
        n_recommendations: Number of similar tracks to return
        method: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        DataFrame with similar tracks and distances
    """
    features = df[['valence', 'arousal']].values
    reference = features[track_idx].reshape(1, -1)
    
    # Use KNN for efficient similarity search
    knn = NearestNeighbors(n_neighbors=n_recommendations + 1, metric=method)
    knn.fit(features)
    
    distances, indices = knn.kneighbors(reference)
    
    # Exclude the reference track itself
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]
    
    similar_tracks = df.iloc[similar_indices].copy()
    similar_tracks['distance'] = similar_distances
    similar_tracks['similarity_score'] = 1 / (1 + similar_distances)  # Convert to similarity
    
    return similar_tracks


def recommend_by_mood(
    df: pd.DataFrame,
    target_valence: float,
    target_arousal: float,
    n_recommendations: int = 10
) -> pd.DataFrame:
    """
    Recommend tracks based on desired mood (valence, arousal).
    
    Args:
        df: DataFrame with track information
        target_valence: Desired valence value
        target_arousal: Desired arousal value
        n_recommendations: Number of recommendations
        
    Returns:
        DataFrame with recommended tracks
    """
    features = df[['valence', 'arousal']].values
    target = np.array([[target_valence, target_arousal]])
    
    # Calculate distances to target mood
    distances = np.array([euclidean(target[0], feat) for feat in features])
    
    # Get top N closest tracks
    top_indices = np.argsort(distances)[:n_recommendations]
    
    recommendations = df.iloc[top_indices].copy()
    recommendations['distance_to_target'] = distances[top_indices]
    recommendations['match_score'] = 1 / (1 + distances[top_indices])
    
    return recommendations


def cluster_by_mood(df: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
    """
    Cluster tracks by emotional similarity using K-Means.
    
    Args:
        df: DataFrame with valence and arousal
        n_clusters: Number of clusters
        
    Returns:
        DataFrame with cluster assignments
    """
    features = df[['valence', 'arousal']].values
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    df_clustered['cluster_center_valence'] = kmeans.cluster_centers_[clusters, 0]
    df_clustered['cluster_center_arousal'] = kmeans.cluster_centers_[clusters, 1]
    
    return df_clustered, kmeans


def analyze_mood_diversity(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze diversity of moods in the dataset.
    
    Args:
        df: DataFrame with valence and arousal
        
    Returns:
        Dictionary with diversity metrics
    """
    features = df[['valence', 'arousal']].values
    
    # Calculate pairwise distances
    n_samples = min(1000, len(df))  # Sample for efficiency
    sample_features = features[np.random.choice(len(features), n_samples, replace=False)]
    
    distances = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances.append(euclidean(sample_features[i], sample_features[j]))
    
    diversity_metrics = {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'coverage_score': np.std(features, axis=0).mean()  # Spread in emotional space
    }
    
    return diversity_metrics


def compute_mood_transition_path(
    df: pd.DataFrame,
    start_valence: float,
    start_arousal: float,
    end_valence: float,
    end_arousal: float,
    n_steps: int = 5
) -> pd.DataFrame:
    """
    Create a mood transition path for playlist generation.
    
    Args:
        df: DataFrame with track information
        start_valence: Starting valence
        start_arousal: Starting arousal
        end_valence: Ending valence
        end_arousal: Ending arousal
        n_steps: Number of intermediate steps
        
    Returns:
        DataFrame with tracks forming a smooth mood transition
    """
    # Create intermediate mood points
    valence_steps = np.linspace(start_valence, end_valence, n_steps)
    arousal_steps = np.linspace(start_arousal, end_arousal, n_steps)
    
    playlist = []
    
    for v, a in zip(valence_steps, arousal_steps):
        # Find closest track to this mood point
        track = recommend_by_mood(df, v, a, n_recommendations=1)
        playlist.append(track.iloc[0])
    
    return pd.DataFrame(playlist)


def get_artist_mood_profile(df: pd.DataFrame, artist_name: str) -> Dict[str, any]:
    """
    Get mood profile for a specific artist.
    
    Args:
        df: DataFrame with track information
        artist_name: Name of the artist
        
    Returns:
        Dictionary with artist mood statistics
    """
    artist_tracks = df[df['artist_name'] == artist_name]
    
    if len(artist_tracks) == 0:
        return None
    
    profile = {
        'artist': artist_name,
        'n_tracks': len(artist_tracks),
        'mean_valence': artist_tracks['valence'].mean(),
        'std_valence': artist_tracks['valence'].std(),
        'mean_arousal': artist_tracks['arousal'].mean(),
        'std_arousal': artist_tracks['arousal'].std(),
        'dominant_quadrant': artist_tracks.apply(
            lambda row: identify_quadrant(row['valence'], row['arousal']), axis=1
        ).mode()[0] if len(artist_tracks) > 0 else None
    }
    
    return profile


def identify_quadrant(valence: float, arousal: float) -> str:
    """
    Identify emotional quadrant.
    
    Args:
        valence: Valence value
        arousal: Arousal value
        
    Returns:
        Quadrant name
    """
    if valence >= 0 and arousal >= 0:
        return 'Happy/Excited'
    elif valence < 0 and arousal >= 0:
        return 'Angry/Tense'
    elif valence < 0 and arousal < 0:
        return 'Sad/Depressed'
    else:
        return 'Calm/Relaxed'


def compute_recommendation_coverage(df: pd.DataFrame, n_recommendations: int = 10) -> float:
    """
    Compute what percentage of catalog can be recommended.
    
    Args:
        df: DataFrame with tracks
        n_recommendations: Number of recommendations per query
        
    Returns:
        Coverage percentage
    """
    n_samples = min(100, len(df))
    sample_indices = np.random.choice(len(df), n_samples, replace=False)
    
    recommended_tracks = set()
    
    for idx in sample_indices:
        similar = find_similar_tracks(df, idx, n_recommendations)
        recommended_tracks.update(similar.index)
    
    coverage = len(recommended_tracks) / len(df) * 100
    return coverage
