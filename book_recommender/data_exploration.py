import kagglehub

path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")

print("Path to dataset files:", path)

import pandas as pd

# Load the dataset
books = pd.read_csv(f"{path}/books.csv")

# Perform data pre-processing
print(books.isna().sum() / len(books) * 100)

# Plot missing data
import seaborn as sns
import matplotlib.pyplot as plt
ax = plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)

# This plot gives a really interesting picture. As we saw, subtitle is mostly missing. We can see that categories, thumbnails, and descriptions are missing but are mostly looking random.
# For the ones at the bottom, average rating, num_pages and rating_count, there is actually a clear pattern here. Observations where one of them missing have all the missing.
# What can be concluded from here is that these are probability from another dataset and that dataset did not contain all the books in the larger dataset.
# This could introduce potential bias because maybe the books that are missing are missing because they are newer or they have other characteristics (maybe they are better rated or worse rated)

# Next, we are going to take a look at the books where the descriptions are missing.

import numpy as np
from datetime import datetime
books['missing_description'] = np.where(books['description'].isna(), 1, 0)
books['age_of_books'] = datetime.today().year - books['published_year']

# Define columns of interest for correlation analysis
columns_of_interest = ['num_pages', 'age_of_books', 'missing_description', 'average_rating']

# Calculate the correlation matrix using Spearman's rank correlation (more appropriate for non-continuous variables)
correlation_matrix = books[columns_of_interest].corr(method='spearman')

# Plot correlation heatmap
plt.figure(figsize=(8,6))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Spearman Correlation'})
heatmap.set_title('Correlation heatmap')
plt.show()

# From looking at the correlation heatmap, there does not seem to be an inherent pattern of missingness of book descriptions within these dimensions. 
# What this means is that if there had been a high correlation with missing description, for instance num of pages and missing desc then if we had dropped the description, 
# then the recommender may have been biased towards longer or shorter books. Same with the age of the book or the average values.

# As there is no inherent pattern to the missingness, it seems to be safe to delete the observations with missing values.

books_v1 = books[~(books['description'].isna()) & ~(books['num_pages'].isna()) & ~(books['average_rating'].isna()) & ~(books['published_year'].isna())]

# Next, let's take a look at the categories column
print(books_v1['categories'].value_counts())  # Lots of categories.

# Check categories with less than 5 counts
books_v1['categories'].value_counts().reset_index().sort_values('count', ascending=False)  # Lots of categories with less than 5 counts

# Ideally, we would like to merge the categories. This will be done using the Large Language models.

# Refine the descriptions further. Notice that there are some descriptions that are just one word. Those would not be very helpful.
books_v1['words_in_description'] = books_v1['description'].apply(lambda x: len(x.split()))

# Looks like meaningful descriptions usually have more than 25 words
books_v1.loc[books_v1['words_in_description'].between(34, 54), 'description']

# Filter out books with descriptions that are too short
books_v2 = books_v1[books_v1['words_in_description'] > 25]

# Since a lot of subtitles are missing, it would be a good idea to combine the title and subtitle
books_v2['title&subtitle'] = np.where(books_v2['subtitle'].isna(), books_v2['title'], books_v2[['title', 'subtitle']].astype(str).agg(": ".join, axis=1))

# Combine ISBN and description into tagged_description
books_v2['tagged_description'] = books_v2[['isbn13', 'description']].astype(str).agg(" ".join, axis=1)

# Drop unnecessary columns and save the cleaned data
books_v2.drop(['subtitle', 'missing_description', 'age_of_books', 'words_in_description'], axis=1).to_csv("books_cleaned.csv", index=False)
