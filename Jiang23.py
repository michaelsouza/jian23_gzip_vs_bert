# %% [markdown]
# Jiang, Zhiying, et al. "“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors." Findings of the Association for Computational Linguistics: ACL 2023. 2023.
# 
# https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
# 
# https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis?utm_medium=social&utm_campaign=kaggle-dataset-share&utm_source=twitter 
# 
# https://pytorch.org/text/stable/datasets.html
# 
# https://www.geeksforgeeks.org/sentiment-classification-using-bert/
# 
# https://github.com/Sentdex/Simple-kNN-Gzip
# 
# https://kenschutte.com/gzip-knn-paper/
# 
# https://kenschutte.com/gzip-knn-paper2/
# 
# https://chat.openai.com/c/a800eb56-29ff-424d-99f9-9ff9bca61adf

# %%
import zipfile
import os

# Path to your zip file
zip_path = 'datasets/movie.zip'
print(zip_path)

# Directory where you want to extract the contents, change as needed
cwd = os.getcwd()
print(cwd)

# Create the directory if it doesn't exist
os.makedirs(cwd, exist_ok=True)

# Open the zip file in read mode
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all the contents into the directory specified
    zip_ref.extractall(cwd)

print(f"Files extracted to {cwd}")


# %%
import os
import pandas as pd

df = pd.read_csv(os.path.join(cwd, 'movie.csv'))

df.rename(columns={'text': 'review'}, inplace=True)

df

# %%
review_counts = df['review'].value_counts()

print(f"Number of text entries: {len(review_counts)}")

def sentiment_word(row):
    return 'positive' if row['label'] == 1 else 'negative'

df['sentiment'] = df.apply(sentiment_word, axis=1)

# Count the number of occurrences of each sentiment (positive/negative)
sentiment_counts = df['sentiment'].value_counts()

print(f"Number of sentiment entries: {len(sentiment_counts)}")

df[['review','sentiment']].groupby('sentiment').count()

# %%
# Create a new column in the DataFrame that contains the length of each review
df['review_length'] = df['review'].apply(len)

# Display the first few rows of the DataFrame to confirm the new column
df.head()

# %%
import matplotlib.pyplot as plt

# Create histograms for review lengths, differentiated by sentiment
plt.figure(figsize=(6, 4))

# Plot for positive reviews
plt.hist(df[df['sentiment'] == 'positive']['review_length'], bins=50, alpha=0.5, label='Positive')

# Plot for negative reviews
plt.hist(df[df['sentiment'] == 'negative']['review_length'], bins=50, alpha=0.5, label='Negative')

plt.xlabel('Review Length (Number of Characters)')
plt.ylabel('Frequency')
plt.title('Histogram of Review Lengths by Sentiment')
plt.legend()

plt.show()

# %%
from scipy.stats import mannwhitneyu
import seaborn as sns

# Mann-Whitney U Test
stat, p = mannwhitneyu(df[df['sentiment'] == 'positive']['review_length'],
                       df[df['sentiment'] == 'negative']['review_length'])

print(f"Mann-Whitney U Test:\nStatistic: {stat}, P-value: {p}\n")

# Interpretation of the test result
alpha = 0.05
if p > alpha:
    print("No significant difference in review lengths (fail to reject H0)")
else:
    print("Significant difference in review lengths (reject H0)")

# Box Plot Visualization
plt.figure(figsize=(6, 4))
sns.boxplot(x='sentiment', y='review_length', data=df)
plt.title('Box Plot of Review Lengths by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Review Length (Number of Characters)')
plt.show()


# %%
from sklearn.model_selection import train_test_split

# Splitting the dataset into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Checking the size of each set
print(f"Training set size: {train_df.shape[0]} reviews")
print(f"Test set size: {test_df.shape[0]} reviews")

# %%
import multiprocessing

# Check the number of available CPU cores for parallelization
cpu_cores = multiprocessing.cpu_count()

cpu_cores

# %%
# Taking a small chunk of the train_df dataset for a test of concept
small_chunk_df = train_df.sample(n=1000, random_state=42)

small_chunk_df.head()

# %%
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import zlib

# Taking a very small chunk for demonstration purposes
tiny_chunk_train_df = train_df.sample(n=10, random_state=42)

def ncd(x, y):
    """
    Computes the Normalized Compression Distance between two strings.
    """
    x_len = len(zlib.compress(x.encode()))
    y_len = len(zlib.compress(y.encode()))
    xy_len = len(zlib.compress((" ".join([x, y])).encode()))
    return (xy_len - min(x_len, y_len)) / max(x_len, y_len)

def compute_ncd(pair):
    """
    Computes the NCD for a pair of reviews.
    Unpacks a tuple containing indices and reviews.
    """
    _, review1, _, review2 = pair
    ncd_value = ncd(review1, review2)
    return ncd_value

def parallel_ncd_matrix(df):
    """
    Constructs the NCD matrix in parallel.
    Generates all possible pairs of reviews from the dataframe,
    computes NCD for each pair, and fills the matrix with these values.
    """
    # Generate all pairs of indices and reviews from the input DataFrame
    pairs = [((i, row1['review'], j, row2['review'])) for i, row1 in df.iterrows() for j, row2 in df.iterrows()]

    # Initialize an empty NCD matrix
    NCD_matrix = np.zeros((df.shape[0], df.shape[0]))

    with ProcessPoolExecutor() as executor:
        # Compute NCD for all pairs using parallel processing
        ncd_values = list(executor.map(compute_ncd, pairs))

        # Fill in the NCD matrix with the computed values
        for idx, ncd_value in enumerate(ncd_values):
            i, j = idx // df.shape[0], idx % df.shape[0]
            NCD_matrix[i, j] = ncd_value

    return NCD_matrix

# Add a feature column calld ncd for each review
def add_ncd_feature(df):
    """
    Adds a new feature column called 'ncd' to the input DataFrame.
    The 'ncd' column contains the NCD values for each review pair.
    """
    # Compute the NCD matrix
    NCD_matrix = parallel_ncd_matrix(df)

    # Add the NCD matrix as a new column to the DataFrame
    df['ncd'] = NCD_matrix.tolist()

    return df

tiny_chunk_train_df = add_ncd_feature(tiny_chunk_train_df)

tiny_chunk_train_df

# %%
tiny_chunk_test_df = test_df.sample(n=10, random_state=42)

def ncd_features(train_df, test_df):
    ncd_mat = np.zeros((test_df.shape[0], train_df.shape[0]))

    for i, testRow in test_df.reset_index().iterrows():
        for j, trainRow in train_df.reset_index().iterrows():
            ncd_mat[i, j] = ncd(trainRow['review'], testRow['review'])

    # Add the NCD matrix as a new column to the test DataFrame
    test_df['ncd'] = ncd_mat.tolist()

    return test_df

tiny_chunk_test_df = ncd_features(tiny_chunk_train_df, tiny_chunk_test_df)

tiny_chunk_test_df

# %%



