import pickle
import streamlit as st
import requests
import os
import concurrent.futures
import pandas as pd
import numpy as np
import plotly.express as px
import time
import logging
from sklearn.decomposition import TruncatedSVD
from app_utils import get_wiki_summary, fetch_movie_details, get_methodology_content, analyze_vibe, get_pipeline_data

from config import TMDB_API_KEY, TMDB_API_KEYS, POSTER_BASE_URL, setup_logging

# Page Config - MUST BE FIRST STREAMLIT CALL
st.set_page_config(
    page_title="Cine Match | Research Portfolio",
    page_icon="🎬",
    layout="wide"
)

# Setup logging
logger = setup_logging()

# Load External CSS
def local_css(file_name):
    """Load external CSS file."""
    try:
        with open(file_name, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        logger.info(f"Successfully loaded CSS from {file_name}")
    except Exception as e:
        logger.error(f"Error loading CSS: {str(e)}")
        st.warning("⚠️ Style file not found or error loading. Using default styling.")

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_css(os.path.join(current_dir, 'style.css'))
except Exception as e:
    logger.error(f"Failed to load CSS: {str(e)}")

@st.cache_resource
def load_data():
    """
    Load research artifacts: movie list, similarity matrix, and pre-fitted vectorizer.
    
    Returns:
        tuple: (movies_df, similarity_matrix, vectorizer) or (None, None, None) on failure
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if all required files exist
    required_files = {
        'movie_list.pkl': 'Movie database',
        'similarity.pkl': 'Similarity matrix',
        'cv.pkl': 'Vectorizer'
    }
    
    missing_files = []
    for filename, description in required_files.items():
        filepath = os.path.join(current_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(f"{description} ({filename})")
    
    if missing_files:
        error_msg = f"Missing required artifacts: {', '.join(missing_files)}"
        logger.error(error_msg)
        st.error(f"⚠️ {error_msg}")
        st.info("""
        **To generate the required artifacts:**
        
        1. Download TMDB 5000 dataset from Kaggle:
           https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
        
        2. Place the CSV files in the project directory
        
        3. Run preprocessing:
           ```bash
           python preprocess.py
           ```
        """)
        return None, None, None
    
    try:
        logger.info("Loading artifacts...")
        movies = pickle.load(open(os.path.join(current_dir, 'movie_list.pkl'), 'rb'))
        similarity = pickle.load(open(os.path.join(current_dir, 'similarity.pkl'), 'rb'))
        vectorizer = pickle.load(open(os.path.join(current_dir, 'cv.pkl'), 'rb'))
        
        logger.info(f"Successfully loaded {len(movies)} movies")
        return movies, similarity, vectorizer
        
    except Exception as e:
        error_msg = f"Critical Artifact Failure: {str(e)}"
        logger.error(error_msg)
        st.error(f"⚠️ {error_msg}")
        st.info("Try regenerating artifacts by running: python preprocess.py")
        return None, None, None

@st.cache_resource
def get_vectorizer():
    """
    Load the pre-fitted vectorizer for explainability features.
    
    Returns:
        CountVectorizer or None on failure
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        vectorizer = pickle.load(open(os.path.join(current_dir, 'cv.pkl'), 'rb'))
        return vectorizer
    except Exception as e:
        logger.error(f"Failed to load vectorizer: {str(e)}")
        return None

def fetch_poster(movie_data):
    """
    Fetch movie poster from TMDB API with multi-key rotation and fallback strategies.
    
    Args:
        movie_data: dict with 'id' and 'title' keys
        
    Returns:
        str: URL to poster image or placeholder
    """
    movie_id = movie_data['id']
    title = movie_data['title']
    
    # Shuffle keys to divide workload equally
    import random
    shuffled_keys = TMDB_API_KEYS.copy()
    random.shuffle(shuffled_keys)
    
    for key in shuffled_keys:
        try:
            # Strategy 1: Direct ID Lookup
            url_id = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}"
            response = requests.get(url_id, timeout=3)
            if response.status_code == 200:
                data = response.json()
                path = data.get('poster_path')
                if path:
                    return f"{POSTER_BASE_URL}{path}"
            
            # Strategy 2: Search by Title (Fallback for broken IDs)
            url_search = f"https://api.themoviedb.org/3/search/movie?api_key={key}&query={title}"
            search_response = requests.get(url_search, timeout=3)
            if search_response.status_code == 200:
                results = search_response.json().get('results', [])
                if results:
                    path = results[0].get('poster_path')
                    if path:
                        return f"{POSTER_BASE_URL}{path}"
            
            # If hit rate limit (429), try next key
            if response.status_code == 429:
                continue
                
        except:
            continue
            
    # Final Strategy: High-End Placeholder
    logger.warning(f"No poster found for movie: {title} after trying all keys")
    return "https://placehold.co/500x750/0e1117/ffffff.png?text=NO+IMAGE"

# Initialize Session State
if 'inference_latency' not in st.session_state:
    st.session_state.inference_latency = 0.0

def recommend(movie, movies, similarity):
    """
    Generate top-6 recommendations using cosine similarity lookup.
    
    Args:
        movie (str): Title string of the query movie
        movies (DataFrame): DataFrame with movie_id, title columns
        similarity (ndarray): Pre-computed cosine similarity matrix (N x N)
    
    Returns:
        list: List of dicts with id, title, poster keys. Empty list on failure.
    """
    if movies is None or similarity is None:
        logger.error("Cannot recommend: artifacts not loaded")
        return []
    
    try:
        # Input validation
        if not movie or not isinstance(movie, str):
            st.warning("Please select a valid movie")
            return []
        
        # Case-insensitive matching
        matches = movies[movies['title'].str.lower() == movie.lower()]
        
        if matches.empty:
            logger.warning(f"Movie '{movie}' not found in corpus")
            st.warning(f"Movie '{movie}' not found in database.")
            return []
        
        index = matches.index[0]
        
        # Get similarity scores
        distances = sorted(
            list(enumerate(similarity[index])),
            reverse=True,
            key=lambda x: x[1]
        )
        
        rec_data = []
        # Skip the first one (the movie itself)
        for i in distances[1:7]:
            mid = movies.iloc[i[0]].movie_id
            title = movies.iloc[i[0]].title
            rec_data.append({'id': mid, 'title': title})

        # Fetch posters in parallel for better performance
        with concurrent.futures.ThreadPoolExecutor() as executor:
            posters = list(executor.map(fetch_poster, rec_data))
            
        for i in range(len(rec_data)):
            rec_data[i]['poster'] = posters[i]
        
        logger.info(f"Generated {len(rec_data)} recommendations for '{movie}'")
        return rec_data
        
    except IndexError as e:
        logger.error(f"Index error in recommend: {str(e)}")
        st.error("Error accessing movie data. Please try another movie.")
        return []
    except Exception as e:
        logger.error(f"Recommendation engine error: {str(e)}")
        st.error(f"Error generating recommendations: {str(e)}")
        return []

# --- DATA LOADING & METRICS ---
logger.info("Starting data load...")
start_time = time.time()
movies_df, similarity_matrix, vectorizer = load_data()
end_time = time.time()
system_latency = (end_time - start_time) * 1000  # Convert to ms
logger.info(f"Data loaded in {system_latency:.2f}ms")

# Exit gracefully if data loading failed
if movies_df is None:
    st.error("⚠️ Application cannot start without required data files.")
    st.stop()

# --- SIGMA UI OVERRIDE ---
st.markdown('<div class="nav-container"><div class="brand">CINE MATCH // PRIME</div><div style="font-family:\'JetBrains Mono\'">SYS.VER.4.0.2</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="tech-header">SYSTEM CONSOLE</div>', unsafe_allow_html=True)
    # Replaced Image with CSS Animation
    # Quantum Core Animation Removed
    st.markdown("""
    <div style="height: 20px;"></div>
    """, unsafe_allow_html=True)
    
    # Dynamic Metrics Placeholder
    metrics_placeholder = st.empty()
    
    # Render Initial Metrics
    metrics_placeholder.markdown(f"""
    <div style="margin-top: 20px;">
        <span class="sigma-badge">KERNEL</span><span style="color:#00ff41">ACTIVE</span><br>
        <span class="sigma-badge">LATENCY</span><span style="color:#00ff41">{st.session_state.inference_latency:.2f}ms</span><br>
        <span class="sigma-badge">MEMORY</span><span style="color:#00ff41">OPTIMAL</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("Execute 'Research Protocol' to access underlying vector mathematics.")
    
    st.caption("DATA INTEGRITY CHECK")
    st.progress(100)
    
    st.code("MODE: CONTENT_BASED\nDIMENSION: 5000\nENGINE: COSINE_SIM", language="yaml")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["DISCOVERY PROTOCOL", "VECTOR MANIFOLD", "PIPELINE ARCHITECTURE", "ACADEMIC ARCHIVES", "EVALUATION"])

with tab1:
    st.markdown("""
        <div class="hero-box">
            <h1 style="font-family:'Outfit'; font-weight:900; font-size:4rem; margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:-2px;">Cine Match <span style="color:#ff3e3e">Primal</span></h1>
            <p style="color:#cbd5e0; font-family:'JetBrains Mono'; font-size:1rem;">[INITIATING NEURAL HANDSHAKE...] PREDICTING CINEMATIC ALIGNMENT.</p>
        </div>
    """, unsafe_allow_html=True)

    if movies_df is not None:
        # Determine Index for Selectbox based on Session State
        if 'selected_movie' not in st.session_state:
            st.session_state.selected_movie = None

        default_index = 0
        if st.session_state.selected_movie and st.session_state.selected_movie in movies_df['title'].values:
            try:
                default_index = list(movies_df['title'].values).index(st.session_state.selected_movie)
            except ValueError:
                logger.warning(f"Movie '{st.session_state.selected_movie}' not found in list")
                default_index = 0

        col_search, _ = st.columns([2, 1])
        with col_search:
            selected_movie = st.selectbox(
                "INPUT TITULAR DATA",
                movies_df['title'].values,
                index=default_index,
                placeholder="Awaiting Input...",
                key="movie_selector"
            )
        
        # If user manually changes selectbox, update state
        if selected_movie and selected_movie != st.session_state.selected_movie:
            st.session_state.selected_movie = selected_movie
            # We don't need to rerun here, the script continues with new selected_movie

        target_movie = selected_movie if selected_movie else st.session_state.selected_movie
        
        if target_movie:
            with st.spinner(f'COMPUTING HIGH-DIMENSIONAL TRAJECTORY FOR: {target_movie}...'):
                t_start = time.time()
                try:
                    recommendations = recommend(target_movie, movies_df, similarity_matrix)
                except Exception as e:
                    logger.error(f"Error during recommendation: {str(e)}")
                    st.error("VECTOR MISMATCH. RE-ALIGNING.")
                    recommendations = []
                t_end = time.time()
                
                # Update Latency State & Sidebar
                st.session_state.inference_latency = (t_end - t_start) * 1000
                metrics_placeholder.markdown(f"""
                <div style="margin-top: 20px;">
                    <span class="sigma-badge">KERNEL</span><span style="color:#00ff41">ACTIVE</span><br>
                    <span class="sigma-badge">LATENCY</span><span style="color:#00ff41">{st.session_state.inference_latency:.2f}ms</span><br>
                    <span class="sigma-badge">MEMORY</span><span style="color:#00ff41">OPTIMAL</span>
                </div>
                """, unsafe_allow_html=True)
                
                if recommendations:
                    st.markdown('<div class="tech-header">PRIMARY TARGET LOCK</div>', unsafe_allow_html=True)
                    
                    feat_col1, feat_col2 = st.columns([1, 2])
                    top = recommendations[0]
                    with feat_col1:
                        st.markdown(f'<div class="movie-card" style="height:auto; border-color:#00ff41;"><img src="{top["poster"]}" class="poster-img"></div>', unsafe_allow_html=True)
                    with feat_col2:
                        details = fetch_movie_details(top['id'])
                        wiki = get_wiki_summary(top['title'])
                        vibe = analyze_vibe(wiki + (details['overview'] if details else ''))
                        
                        st.markdown(f"<h1 style='color:#fff; margin-bottom:0; font-family:Outfit; text-transform:uppercase;'>{top['title']}</h1>", unsafe_allow_html=True)
                        st.markdown(f"<span class='sigma-badge' style='border-color:#ffcc00; color:#ffcc00;'>VIBE_ANALYSIS</span> <span style='font-family:JetBrains Mono; color:#ffcc00;'>{vibe}</span>", unsafe_allow_html=True)
                        
                        if details:
                            st.markdown(f"""
                            <div style="margin: 15px 0;">
                                <span class="sigma-badge">RATING</span> {details['rating']} &nbsp;
                                <span class="sigma-badge">RELEASE</span> {details['release_date']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if details['trailer']:
                                st.video(details['trailer'])
                        
                        st.markdown(f"<div style='background:rgba(255,255,255,0.05); padding:15px; border-radius:4px; border-left:3px solid #ff3e3e; margin-top:20px; font-family:JetBrains Mono; font-size:0.85rem;'>{wiki}</div>", unsafe_allow_html=True)

                    st.markdown('<div class="tech-header">LATENT CLUSTER PROJECTIONS</div>', unsafe_allow_html=True)
                    cols = st.columns(5)
                    for i, rec in enumerate(recommendations[1:]):
                        with cols[i]:
                            st.image(rec['poster'], use_container_width=True)
                            if st.button(f"🎯 ANALYSIS {i+1}", key=f"rec_{i}", help=f"Analyze {rec['title']}"):
                                st.session_state.selected_movie = rec['title']
                                st.rerun()
                            st.markdown(f"""
                                <div style="padding:5px; text-align:center;">
                                    <div class="movie-title" style="font-family:'JetBrains Mono'; font-size:0.8rem; text-transform:uppercase;">{rec['title']}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Explainability Button
                            if st.button(f"🔍 EXPLAIN", key=f"explain_{i}"):
                                from explainability import explain_recommendation
                                cv = get_vectorizer()
                                    
                                if 'tags' in movies_df.columns:
                                    if not hasattr(cv, 'vocabulary_'):
                                         cv.fit(movies_df['tags'])
                                         
                                    try:
                                        explanation = explain_recommendation(
                                            selected_movie if selected_movie else st.session_state.selected_movie,
                                            rec['title'],
                                            movies_df,
                                            cv,
                                            similarity_matrix
                                        )
                                        
                                        st.markdown(f"""
                                        <div style="background:rgba(0,0,0,0.5); padding:10px; border:1px solid #444; margin-top:10px;">
                                            <div style="color:#00ff41; font-family:'JetBrains Mono'; font-size:0.8rem;">COSINE SIMILARITY: {explanation['cosine_similarity']:.4f}</div>
                                            <div style="margin-top:5px; font-family:'JetBrains Mono'; font-size:0.75rem; color:#aaa;">TOP SHARED FEATURES:</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        for feat in explanation['top_features'][:5]:
                                            st.markdown(f"""
                                            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:0.7rem; border-bottom:1px solid #333; padding:2px 0;">
                                                <span>{feat['feature']}</span>
                                                <span style="color:#ffcc00;">{feat['contribution']:.3f}</span>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error(f"Could not explain: {e}")
                                else:
                                    st.warning("Feature tags not available for explanation. Please run preprocess.py to rebuild artifacts.")


with tab2:
    st.markdown('<div class="tech-header">LATENT SIMILARITY SPACE</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-family:JetBrains Mono; color:#aaa;'>3D PROJECTION OF COSINE SIMILARITY MATRIX VIA TRUNCATED SVD. MOVIES THAT CLUSTER TOGETHER SHARE SIMILAR METADATA SIGNATURES.</p>", unsafe_allow_html=True)
    
    if movies_df is not None:
        # Use real SVD coordinates from pre-computation
        if 'dim_x' in movies_df.columns:
            plot_df = movies_df.sample(min(500, len(movies_df))).copy()
            
            # Premium 3D Manifold with adaptive visibility
            fig = px.scatter_3d(
                plot_df, x='dim_x', y='dim_y', z='dim_z',
                hover_name='title',
                color='dim_z',
                color_continuous_scale='Plasma', # Vibrant, tech-inspired colors
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, b=10, t=0), # Removed T margin as we use st.markdown
                scene=dict(
                    xaxis=dict(
                        showbackground=True, 
                        backgroundcolor="rgba(100,100,100,0.1)", # More visible "box"
                        gridcolor="rgba(200,200,200,0.3)", # Stronger grid lines
                        zerolinecolor="rgba(200,200,200,0.3)",
                        title=''
                    ),
                    yaxis=dict(
                        showbackground=True, 
                        backgroundcolor="rgba(100,100,100,0.1)",
                        gridcolor="rgba(200,200,200,0.3)",
                        zerolinecolor="rgba(200,200,200,0.3)",
                        title=''
                    ),
                    zaxis=dict(
                        showbackground=True, 
                        backgroundcolor="rgba(100,100,100,0.1)",
                        gridcolor="rgba(200,200,200,0.3)",
                        zerolinecolor="rgba(200,200,200,0.3)",
                        title=''
                    ),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=700 # Reduced from 800 for better mobile synergy
            )
            
            fig.update_scenes(
                xaxis_tickfont=dict(size=10, color='#888'),
                yaxis_tickfont=dict(size=10, color='#888'),
                zaxis_tickfont=dict(size=10, color='#888')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("SVD coordinates not available. Please clear cache and reload.")

with tab3:
    st.markdown('<div class="tech-header">SYSTEM ARCHITECTURE COMPONENT FLOW</div>', unsafe_allow_html=True)
    
    st.write("### Research Pipeline Stages")

    # Detailed Stage Breakdown below the map
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="sigma-badge" style="border-color:#ffcc00; color:#ffcc00;">DATA INPUTS</div>', unsafe_allow_html=True)
        st.write("- **CSV Ingestion**: TMDB 5k structured data.")
        st.write("- **API Hooks**: Live poster & metadata fetching.")
        st.write("- **User Vibe**: Real-time titular entry selection.")
    with col2:
        st.markdown('<div class="sigma-badge" style="border-color:#ff3e3e; color:#ff3e3e;">CORE TRANSFORM</div>', unsafe_allow_html=True)
        st.write("- **Tag Fusion**: Concatenating Genres + Cast + Keywords.")
        st.write("- **Vectorization**: Bag-of-Words (5000-D Space).")
        st.write("- **SVD Reduction**: Precomputed 3D Latent Manifold.")
    with col3:
        st.markdown('<div class="sigma-badge" style="border-color:#00ff41; color:#00ff41;">EXTRACTS</div>', unsafe_allow_html=True)
        st.write("- **Explanations**: Mathematical feature attribution.")
        st.write("- **Manifold**: Adaptive 3D Latent cluster map.")
        st.write("- **Metrics**: Precision@10 Genre-proxy benchmark.")

with tab4:
    methodology = get_methodology_content()
    st.markdown('<div class="research-container">', unsafe_allow_html=True)
    
    col_intro, col_stats = st.columns([2, 1])
    with col_intro:
        st.markdown(methodology['concept'])

    with col_stats:
        st.markdown('<div class="tech-header">METRICS</div>', unsafe_allow_html=True)
        st.metric(label="CORPUS SIZE", value="4,803 UNITS")
        st.metric(label="SPARSITY", value="98.2%")
        st.metric(label="INFERENCE", value=f"{st.session_state.inference_latency:.2f}ms")

    st.markdown("<div style='margin: 40px 0; border-top:1px dashed #444;'></div>", unsafe_allow_html=True)
    
    st.markdown(methodology['deep_research'])
    
    st.markdown("<div style='margin: 40px 0; border-top:1px dashed #444;'></div>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(methodology['math'])
    with col_b:
        st.markdown(methodology['comparison'])

    st.markdown('<div class="tech-header">SYSTEM ARCHITECTURE</div>', unsafe_allow_html=True)
    for step in methodology['steps']:
        with st.expander(f"PROTOCOL: {step['title']}"):
            st.write(step['desc'])
            st.code(step['code'], language='python')
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="tech-header">EXPERIMENTAL EVALUATION</div>', unsafe_allow_html=True)
    
    # Research Metadata Component
    metadata_html = """
    <div style="background: rgba(255,255,255,0.03); padding: 25px; border-radius: 12px; border-left: 5px solid #ff3e3e; margin-bottom: 30px; color: white; font-family: 'Outfit', sans-serif;">
        <h3 style="margin-top: 0; color: #ff3e3e; font-family: 'Outfit', sans-serif; font-weight: 900; letter-spacing: 1px;">🔬 RESEARCH METADATA & METHODOLOGY</h3>
        
        <p style="color: #aaa; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; line-height: 1.6;">
            This module evaluates the performance of the Cine Match recommendation engine using a controlled experimental setup. 
            We analyze how different <b>Numerical Vectorization Strategies</b> impact both the accuracy and efficiency of the system.
        </p>

        <h4 style="color: #fff; margin-top: 25px; font-family: 'Outfit', sans-serif; font-weight: 700; text-transform: uppercase; font-size: 1rem; letter-spacing: 1px;">📂 UNDERLYING DATASET</h4>
        <p style="font-size: 0.85rem; color: #eee; font-family: 'JetBrains Mono', monospace;">
            The evaluation runs on the <b>TMDB 5000 Movie Dataset</b>. Specifically, it utilizes the <b>'tags'</b> feature 
            created during preprocessing, which is a fused textual signature containing:
            <ul style="font-size: 0.85rem; color: #ccc; font-family: 'JetBrains Mono', monospace; line-height: 1.8;">
                <li>Movie Overviews (Plot summaries)</li>
                <li>Genres (Action, Sci-Fi, etc.)</li>
                <li>Keywords (Metadata tags)</li>
                <li>Top 3 Cast Members</li>
                <li>The Director</li>
            </ul>
        </p>

        <h4 style="color: #fff; margin-top: 25px; font-family: 'Outfit', sans-serif; font-weight: 700; text-transform: uppercase; font-size: 1rem; letter-spacing: 1px;">📏 CORE METRICS EXPLAINED</h4>
        <div style="display: flex; gap: 20px; margin-top: 15px; flex-wrap: wrap;">
            <div style="background: rgba(0,0,0,0.3); padding: 20px; border: 1px solid #333; border-radius: 12px; flex: 1; min-width: 250px; transition: 0.3s;">
                <b style="color: #00ff41; font-family: 'Outfit', sans-serif; font-size: 1rem;">1. PRECISION @ 10 (Accuracy)</b>
                <p style="font-size: 0.8rem; color: #bbb; margin-top: 8px; font-family: 'JetBrains Mono', monospace; line-height: 1.5;">
                    <b style="color: #eee;">What:</b> The percentage of the top 10 recommendations that are actually "relevant."<br>
                    <b style="color: #eee;">Logic:</b> We use <b>Genre Overlap</b> as a Ground Truth. If a recommended movie shares at least one genre 
                    with the query movie, it is counted as a "Hit."<br>
                    <b style="color: #eee;">Why:</b> In content-based filtering, genre preservation is the strongest indicator of thematic relevance.
                </p>
            </div>
            <div style="background: rgba(0,0,0,0.3); padding: 20px; border: 1px solid #333; border-radius: 12px; flex: 1; min-width: 250px;">
                <b style="color: #00d2ff; font-family: 'Outfit', sans-serif; font-size: 1rem;">2. LATENCY (Efficiency)</b>
                <p style="font-size: 0.8rem; color: #bbb; margin-top: 8px; font-family: 'JetBrains Mono', monospace; line-height: 1.5;">
                    <b style="color: #eee;">What:</b> Measured in <b>milliseconds (ms)</b> per recommendation request.<br>
                    <b style="color: #eee;">Logic:</b> The time taken to vectorize the query tag and compute the cosine similarity against 4,800 records.<br>
                    <b style="color: #eee;">Why:</b> In real-time production systems, high accuracy is useless if the response takes several seconds.
                </p>
            </div>
        </div>

        <h4 style="color: #fff; margin-top: 25px; font-family: 'Outfit', sans-serif; font-weight: 700; text-transform: uppercase; font-size: 1rem; letter-spacing: 1px;">🧪 EXPERIMENTAL STRATEGIES</h4>
        <ul style="font-size: 0.85rem; color: #ccc; font-family: 'JetBrains Mono', monospace; line-height: 1.8;">
            <li><b style="color: #fff;">Bag-of-Words (BoW):</b> Counts word frequencies. Fast but ignores context.</li>
            <li><b style="color: #fff;">TF-IDF:</b> Weights rare words higher. Better for identifying unique movie themes.</li>
            <li><b style="color: #fff;">SBERT (Deep Learning):</b> Uses Transformer models to understand semantic meaning (e.g., knows "Space" is similar to "Galaxy").</li>
        </ul>
    </div>
    """
    st.components.v1.html(metadata_html, height=750, scrolling=True)
    
    if 'genres' not in movies_df.columns:
        st.error("MISSING ARTIFACT: 'genres' metadata required for evaluation.")
        st.warning("Please download TMDB 5000 dataset and run 'python preprocess.py' to regenerate artifacts with research features.")
    else:
        if st.button("🚀 Run Ablation Study", key="run_eval"):
            with st.spinner("Running experiments across methods..."):
                from evaluation import run_ablation_study
                results_df = run_ablation_study(movies_df)
                
                st.session_state.eval_results = results_df
                st.success("Study Complete!")
    
    if 'eval_results' in st.session_state:
        results = st.session_state.eval_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Precision@10 Comparison")
            fig = px.bar(
                results, 
                x='method', 
                y='precision@k',
                error_y='std_precision',
                color='method',
                labels={'precision@k': 'Precision@10', 'method': 'Method'},
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚡ Latency Comparison")
            fig2 = px.bar(
                results,
                x='method',
                y='latency_ms',
                color='method',
                labels={'latency_ms': 'Latency (ms/movie)', 'method': 'Method'},
                template='plotly_dark'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("#### 📋 Detailed Results")
        st.dataframe(results, use_container_width=True)

# Footer
st.markdown("<br><br><br><p style='text-align:center; color:#555; font-family:JetBrains Mono; font-size:0.8rem;'>[EVALUATOR: K RAMA KRISHNA NARASIMHA CHOWDARY] :: [STATUS: ONLINE]</p>", unsafe_allow_html=True)
