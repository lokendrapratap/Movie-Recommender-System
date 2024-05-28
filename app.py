import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.title("Movie Recommender System")

# Load data
df_cred = pd.read_csv("tmdb_5000_credits.csv")
df_mov = pd.read_csv("tmdb_5000_movies.csv")

# Check the size of data sets
st.write(f"Credits Dataset: {df_cred.shape}")
st.write(f"Movies Dataset: {df_mov.shape}")

# Merge the two DataFrames together
df_cred.rename(columns={'movie_id':'id'}, inplace=True)
df = df_cred.merge(df_mov, on='id')

# Drop null overviews
df.dropna(subset=['overview'], inplace=True)

# Filter out target columns
df = df[['id', 'title_x', 'genres', 'overview', 'cast', 'crew']]

# Generate corpus
def generate_corpus(row):
    genre = ' '.join([i['name'] for i in eval(row['genres'])])
    cast = ' '.join([i['name'] for i in eval(row['cast'])[:3]])
    crew = ' '.join(list(set([i['name'] for i in eval(row['crew']) if i['job']=='Director' or i['job']=='Producer'])))
    corpus = row['overview'] + " " + genre + " " + cast + " " + crew
    return corpus

df['corpus'] = df.apply(generate_corpus, axis=1)

# Compute cosine similarity
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(df['corpus'])
similarity_matrix = cosine_similarity(count_matrix, count_matrix)

# Create a DataFrame using the cosine similarity matrix
similarity_df = pd.DataFrame(similarity_matrix, columns=df['title_x'], index=df['title_x'])

# Function to get recommendations
def get_recommendations(title, similarity_df):
    if title not in similarity_df.columns:
        return []
    index = similarity_df.columns.get_loc(title)
    similarities = list(enumerate(similarity_df.iloc[index]))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[1:11]

# Get user input
title_input = st.text_input("Enter a movie title:")
if title_input:
    title = title_input
    recommendations = get_recommendations(title, similarity_df)
    if recommendations:
        st.write("Recommendations:")
        for i, (index, similarity) in enumerate(recommendations):
            st.write(f"{i+1}. {df.iloc[index]['title_x']} - Similarity: {similarity:.2f}")
    else:
        st.write("Movie not found. Please enter a valid movie title.")
