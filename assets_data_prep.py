import pickle
import os
import inspect
import pandas as pd
import lzma

def prepare_data(df):
    """
    Preprocesses the movie dataset by applying historical statistical mappings 
    and feature engineering. Designed to be self-contained for inference.
    """

    # --- 1. Helper for genre standardization to ensure consistency ---
    def standardize_genres(genre_val):
        if isinstance(genre_val, str):
            # Clean string and split into a list
            cleaned = genre_val.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
            return [g.strip() for g in cleaned.split(',') if g.strip()]
        elif isinstance(genre_val, list):
            return [str(g).strip() for g in genre_val if str(g).strip()]
        return []

    # --- 2. Prevent data leakage by detecting if we are in TRAIN or TEST ---
    frame = inspect.currentframe().f_back
    variable_name = next((name for name, val in frame.f_locals.items() if val is df), None)
    is_train = (variable_name == 'features_train')
    print(f"--- Running {'TRAIN' if is_train else 'TEST'} process (Source: {variable_name}) ---")

    # --- 3. Load precomputed statistical mappings ---
    global mappings
    if 'mappings' not in globals():
        if os.path.exists('mappings.pkl.xz'):
            with lzma.open('mappings.pkl.xz', 'rb') as f:
                mappings = pickle.load(f)
        else:
            with open('mappings.pkl', 'rb') as f:
                mappings = pickle.load(f)

    data = df.copy()

    # --- 4. Input validation and basic data cleaning ---
    required_cols = ['startYear', 'genres', 'runtimeMinutes']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    if 'tconst' not in data.columns:
        data['tconst'] = 'tt_dummy'

    # Convert types and handle missing genre strings
    data['startYear'] = pd.to_numeric(data['startYear'], errors='coerce')
    data['runtimeMinutes'] = pd.to_numeric(data['runtimeMinutes'], errors='coerce')
    data['genres'] = data['genres'].replace(['\\N', '\\\\N', ''], 'Unknown').fillna('Unknown')

    # --- 5. Feature Engineering: Imputation and decade calculation ---
    data['runtimeMinutes'] = data['runtimeMinutes'].fillna(mappings['train_runtime_median'])
    data['decade'] = (data['startYear'] // 10 * 10).astype(int)
    data['genres_list'] = data['genres'].apply(standardize_genres)

    # One-Hot Encoding for genres
    for genre in mappings['unique_genres']:
        data[f'genre_{genre}'] = data['genres_list'].apply(lambda x: 1 if genre in x else 0)

    # --- 6. Historical Feature Engineering (lookup based) ---
    def get_hist_features(row):
        genres = row['genres_list']
        year = row['startYear']
        ratings = [mappings['hist_dict'].get((year, g), 0.0) for g in genres]
        runtimes = [mappings['hist_runtime_dict'].get((year, g), 0.0) for g in genres]
        valid_r = [r for r in ratings if r > 0]
        valid_run = [r for r in runtimes if r > 0]
        return (sum(valid_r)/len(valid_r) if valid_r else 0.0), \
               (sum(valid_run)/len(valid_run) if valid_run else 0.0)

    features = data.apply(get_hist_features, axis=1, result_type='expand')
    data[['hist_genre_rating', 'hist_genre_runtime']] = features
    data['runtime_deviation'] = data['runtimeMinutes'] - data['hist_genre_runtime']

    # --- 7. Director-based Feature Engineering (Stable merge logic) ---
    if 'crew_df' in mappings:
        data = data.merge(
            mappings['crew_df'][['tconst', 'directors']],
            on='tconst',
            how='left',
            suffixes=('', '_crew')
        )

        if 'directors_crew' in data.columns:
            # TRAIN mode: use crew_df if column is missing
            if 'directors' not in df.columns:
                data['directors'] = data['directors_crew']
            else:
                # TEST mode: fill missing values with crew_df
                mask = data['directors_crew'].notna() & (data['directors_crew'] != '')
                data.loc[mask, 'directors'] = data.loc[mask, 'directors_crew']
            data = data.drop(columns=['directors_crew'])

    # Handle multiple directors and aggregate performance history
    data['directors_clean'] = data['directors'].fillna('').str.split(',')
    exploded = data.explode('directors_clean')
    exploded = exploded[exploded['directors_clean'] != '']

    exploded = exploded.merge(
        mappings['yearly'][['directors', 'startYear', 'director_hist_year']],
        left_on=['directors_clean', 'startYear'],
        right_on=['directors', 'startYear'],
        how='left'
    )

    # Groupby to get mean historical features per movie
    director_features = exploded.groupby('tconst')['director_hist_year'].mean().reset_index()
    director_features.columns = ['tconst', 'director_hist']

    data = data.merge(director_features, on='tconst', how='left')
    data['director_hist'] = data['director_hist'].fillna(mappings['global_mean'])

    # Flag for multiple directors
    data['has_multiple_directors'] = data['directors'].fillna('').str.split(',').apply(lambda x: 1 if len(x) > 1 else 0)

    # Global director statistics lookup
    stats = data['tconst'].map(mappings['tconst_dir_map']).apply(
        lambda x: x if isinstance(x, tuple) else (0.0, 0)
    )
    stats_df = pd.DataFrame(
        stats.tolist(),
        index=data.index,
        columns=['director_std_global', 'director_known']
    )
    data = pd.concat([data, stats_df], axis=1)

    # --- 8. Final cleanup: Drop auxiliary columns ---
    cols_to_drop = [
        'startYear', 'genres', 'averageRating', 'directors',
        'directors_clean', 'genres_list',
        'lead_actors_ids', 'Language', 'Country',
        'genre_\\N', 'genre_\\\\N', 'tconst', 'primaryTitle',
        'numVotes', 'BoxOffice', 'budget', 'plot'
    ]

    final_df = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    return final_df