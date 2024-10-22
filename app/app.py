"""
Spotify Track Popularity Predictor - Streamlit App
Using native Streamlit components for better integration and reliability
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from typing import Tuple
import time

# Page configuration
st.set_page_config(
    page_title="Spotify Track Popularity Predictor",
    page_icon=":musical_note:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constants and Paths
CURRENT_DIR = os.getcwd()
MODEL_PATH = os.path.join(CURRENT_DIR, 'spotify_popularity_model.joblib')
SCALER_PATH = os.path.join(CURRENT_DIR, 'spotify_scaler.joblib')

# Cache the model and scaler loading
@st.cache_resource
def load_model_and_scaler():
    """Load the pre-trained model and scaler"""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def create_feature_plot(features: dict) -> plt.Figure:
    """Create a radar plot of track features"""
    # Use a built-in style
    plt.style.use('default')
    
    # Create figure with light background
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('#F0F2F6')
    ax.set_facecolor('#FFFFFF')
    
    # Select features for plot
    features_for_plot = {k: v for k, v in features.items() 
                        if k not in ['tempo', 'duration_ms', 'loudness']}
    
    # Prepare data
    angles = np.linspace(0, 2*np.pi, len(features_for_plot), endpoint=False)
    values = list(features_for_plot.values())
    values += values[:1]
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create plot with custom colors
    ax.plot(angles, values, 'o-', linewidth=2, 
           color='#1DB954',  # Spotify green
           label='Features')
    ax.fill(angles, values, alpha=0.25, color='#1DB954')
    
    # Customize ticks and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_for_plot.keys())
    
    # Set limits and grid style
    ax.set_ylim(0, 1)
    ax.grid(True, color='gray', alpha=0.3)
    
    # Add title
    plt.title('Track Feature Analysis', pad=20, fontsize=12)
    
    return fig

def main():
    # Title and Introduction
    st.title('Spotify Track Popularity Predictor')
    
    # Information expander
    with st.expander("About this app", expanded=False):
        st.write("""
        This application helps predict a track's popularity on Spotify based on its audio features. 
        Use the sliders below to adjust various track characteristics and see how they affect the predicted popularity.
        """)
        st.info("""
        The prediction is based on machine learning analysis of Spotify track data.
        The model considers both audio features and artist metrics to make its prediction.
        """)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("Failed to load model. Please check the model files.")
        return
    
    st.divider()
    
    # Main form for feature input
    with st.form("track_features_form"):
        st.header("Track Features")
        
        # Audio Features Section
        st.subheader("Audio Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            danceability = st.slider(
                "Danceability",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="How suitable the track is for dancing"
            )
            
            energy = st.slider(
                "Energy",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Intensity and activity level"
            )
            
            speechiness = st.slider(
                "Speechiness",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                help="Presence of spoken words"
            )
            
            acousticness = st.slider(
                "Acousticness",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Amount of acoustic sound"
            )
        
        with col2:
            instrumentalness = st.slider(
                "Instrumentalness",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Amount of instrumental content"
            )
            
            liveness = st.slider(
                "Liveness",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Presence of live audience"
            )
            
            valence = st.slider(
                "Valence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Musical positiveness"
            )
            
            loudness = st.slider(
                "Loudness (dB)",
                min_value=-60.0,
                max_value=0.0,
                value=-10.0,
                help="Overall loudness"
            )
        
        st.divider()
        
        # Track Metrics Section
        st.subheader("Track Metrics")
        
        col3, col4 = st.columns(2)
        
        with col3:
            tempo = st.slider(
                "Tempo (BPM)",
                min_value=50.0,
                max_value=200.0,
                value=120.0,
                help="Beats per minute"
            )
        
        with col4:
            duration_ms = st.slider(
                "Duration (ms)",
                min_value=30000,
                max_value=600000,
                value=200000,
                step=1000,
                help="Track duration in milliseconds"
            )
        
        st.divider()
        
        # Artist Metrics Section
        st.subheader("Artist Metrics")
        artist_popularity = st.slider(
            "Artist Popularity",
            min_value=0,
            max_value=100,
            value=50,
            help="Current popularity of the artist"
        )
        
        # Submit button
        submitted = st.form_submit_button("Predict Popularity")
    
    # Handle prediction when form is submitted
    if submitted:
        # Prepare features
        features = {
            'danceability': danceability,
            'energy': energy,
            'loudness': loudness,
            'speechiness': speechiness,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'liveness': liveness,
            'valence': valence,
            'tempo': tempo,
            'duration_ms': duration_ms,
            'artist_popularity': artist_popularity
        }
        
        # Show progress
        with st.spinner("Analyzing track features..."):
            progress_bar = st.progress(0)
            
            # Simulate processing
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Make prediction
            X = np.array(list(features.values())).reshape(1, -1)
            X_scaled = scaler.transform(X)
            prediction = model.predict_proba(X_scaled)[0]
            popularity_score = prediction[1] * 100
        
        st.divider()
        
        # Display Results
        st.header("Prediction Results")
        
        # Metrics
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.metric(
                "Popularity Score",
                f"{popularity_score:.1f}%",
                f"{popularity_score - 50:.1f}%"
            )
        
        with col_metrics2:
            st.metric(
                "Energy Level",
                f"{energy * 100:.1f}%"
            )
        
        with col_metrics3:
            st.metric(
                "Artist Popularity",
                f"{artist_popularity}%"
            )
        
        # Results visualization
        col_viz, col_info = st.columns(2)
        
        with col_viz:
            st.subheader("Feature Analysis")
            fig = create_feature_plot(features)
            st.pyplot(fig)
        
        with col_info:
            st.subheader("Prediction Analysis")
            
            if popularity_score > 70:
                st.success("This track shows strong potential for popularity!")
            else:
                st.warning("This track might need some adjustments to increase its popularity.")
            
            st.info("Key Factors Affecting Popularity:")
            st.write("1. Artist Popularity")
            st.write("2. Danceability and Energy")
            st.write("3. Valence (Musical Positiveness)")
            st.write("4. Production Quality (Loudness)")

if __name__ == "__main__":
    main()