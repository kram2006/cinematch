import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import logging

# Setup logging
logger = logging.getLogger('CineMatch.explainability')

def explain_recommendation(query_movie, recommended_movie, movies_df, vectorizer, similarity_matrix):
    """
    Explain why a movie was recommended by showing top contributing features.
    
    Args:
        query_movie (str): The query movie title
        recommended_movie (str): The recommended movie title
        movies_df (DataFrame): Movies dataframe with 'title' and 'tags' columns
        vectorizer (CountVectorizer): Fitted vectorizer
        similarity_matrix (ndarray): Pre-computed similarity matrix
    
    Returns:
        dict: Explanation with cosine_score, top_features, and movie titles
    """
    try:
        # Find movie indices
        query_matches = movies_df[movies_df['title'] == query_movie]
        rec_matches = movies_df[movies_df['title'] == recommended_movie]
        
        if query_matches.empty:
            logger.error(f"Query movie not found: {query_movie}")
            raise ValueError(f"Query movie '{query_movie}' not found in database")
        
        if rec_matches.empty:
            logger.error(f"Recommended movie not found: {recommended_movie}")
            raise ValueError(f"Recommended movie '{recommended_movie}' not found in database")
        
        query_idx = query_matches.index[0]
        rec_idx = rec_matches.index[0]
        
        # Cosine similarity score
        cosine_score = similarity_matrix[query_idx, rec_idx]
        
        # Get feature vectors
        query_vec = vectorizer.transform([movies_df.iloc[query_idx]['tags']])
        rec_vec = vectorizer.transform([movies_df.iloc[rec_idx]['tags']])
        
        # Find overlapping features
        feature_names = vectorizer.get_feature_names_out()
        query_features = query_vec.toarray()[0]
        rec_features = rec_vec.toarray()[0]
        
        # Element-wise product to find shared features
        overlap = query_features * rec_features
        top_indices = np.argsort(overlap)[-10:][::-1]  # Top 10 shared features
        
        top_features = [
            {
                'feature': feature_names[idx],
                'query_count': int(query_features[idx]),
                'rec_count': int(rec_features[idx]),
                'contribution': float(overlap[idx])
            }
            for idx in top_indices if overlap[idx] > 0
        ]
        
        logger.info(f"Generated explanation for {query_movie} -> {recommended_movie}")
        
        return {
            'cosine_similarity': float(cosine_score),
            'top_features': top_features,
            'query_movie': query_movie,
            'recommended_movie': recommended_movie
        }
        
    except ValueError as e:
        logger.error(f"Value error in explain_recommendation: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise
