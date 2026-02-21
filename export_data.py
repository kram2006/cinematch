"""
Export pickle data to lightweight JSON for GitHub Pages static site.
Includes pre-computed shared features for explainability.
Run once: python export_data.py
"""
import pickle
import json
import os
import numpy as np

print("[*] Loading pickle artifacts...")

current_dir = os.path.dirname(os.path.abspath(__file__))

movies = pickle.load(open(os.path.join(current_dir, 'movie_list.pkl'), 'rb'))
similarity = pickle.load(open(os.path.join(current_dir, 'similarity.pkl'), 'rb'))
cv = pickle.load(open(os.path.join(current_dir, 'cv.pkl'), 'rb'))

print(f"[OK] Loaded {len(movies)} movies, similarity shape: {similarity.shape}")

# Create data directory
data_dir = os.path.join(current_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# Pre-compute vectorized tags for explainability
print("[*] Vectorizing tags for feature extraction...")
tags_list = movies['tags'].tolist()
vectors = cv.transform(tags_list)
feature_names = cv.get_feature_names_out()
print(f"[OK] Vectorized {vectors.shape[0]} movies x {vectors.shape[1]} features")

# 1. Export movie list (id, title, genres)
movies_list = []
for idx, row in movies.iterrows():
    entry = {
        'id': int(row['movie_id']),
        'title': row['title'],
    }
    if 'genres' in movies.columns:
        genres = row['genres']
        if isinstance(genres, list):
            entry['genres'] = genres
        else:
            entry['genres'] = []
    movies_list.append(entry)

with open(os.path.join(data_dir, 'movies.json'), 'w', encoding='utf-8') as f:
    json.dump(movies_list, f, ensure_ascii=False)
print(f"[OK] Exported movies.json ({len(movies_list)} movies)")

# 2. Export top 10 recommendations per movie WITH shared features
print("[*] Computing recommendations with shared features (this takes a moment)...")
recommendations = {}
total = len(movies)
for idx in range(total):
    if idx % 500 == 0:
        print(f"     Processing {idx}/{total}...")
    
    distances = list(enumerate(similarity[idx]))
    distances.sort(key=lambda x: x[1], reverse=True)
    
    title = movies.iloc[idx]['title']
    query_vec = vectors[idx].toarray()[0]
    
    recs = []
    for i, score in distances[1:11]:  # Top 10, skip self
        rec_vec = vectors[i].toarray()[0]
        
        # Compute shared features (element-wise product)
        overlap = query_vec * rec_vec
        top_indices = np.argsort(overlap)[-5:][::-1]  # Top 5 shared features
        
        features = []
        for fi in top_indices:
            if overlap[fi] > 0:
                features.append({
                    'f': feature_names[fi],     # feature name
                    'c': round(float(overlap[fi]), 3)  # contribution
                })
        
        recs.append({
            'id': int(movies.iloc[i]['movie_id']),
            'title': movies.iloc[i]['title'],
            'score': round(float(score), 4),
            'features': features
        })
    recommendations[title] = recs

with open(os.path.join(data_dir, 'recommendations.json'), 'w', encoding='utf-8') as f:
    json.dump(recommendations, f, ensure_ascii=False)
print(f"[OK] Exported recommendations.json ({len(recommendations)} movies)")

# 3. Export 3D manifold coordinates
if 'dim_x' in movies.columns:
    manifold = []
    for idx, row in movies.iterrows():
        manifold.append({
            'title': row['title'],
            'x': round(float(row['dim_x']), 4),
            'y': round(float(row['dim_y']), 4),
            'z': round(float(row['dim_z']), 4)
        })
    with open(os.path.join(data_dir, 'manifold.json'), 'w', encoding='utf-8') as f:
        json.dump(manifold, f, ensure_ascii=False)
    print(f"[OK] Exported manifold.json ({len(manifold)} points)")
else:
    print("[WARN] No SVD coordinates found, skipping manifold export")

# Print file sizes
print("\nFile sizes:")
for fname in os.listdir(data_dir):
    fpath = os.path.join(data_dir, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"   {fname}: {size_mb:.2f} MB")

print("\n[DONE] Export complete! JSON files are in the 'data/' directory.")
