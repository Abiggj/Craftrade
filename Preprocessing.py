import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set folder path and gather all file addresses
folder_path = "./news"
file_addresses = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)

# Function to keep the most important 250 words using TF-IDF
def keep_top_words(texts, num_words=250):
    vectorizer = TfidfVectorizer(max_features=num_words)
    X = vectorizer.fit_transform(texts)
    important_words = vectorizer.get_feature_names_out()
    result_texts = []
    for row in X.toarray():
        result_texts.append(" ".join([word for word, score in zip(important_words, row) if score > 0]))
    return result_texts

dates, news = [], []
for filename in file_addresses:
    with open(filename, 'r', encoding="utf8") as file:
        print(filename, "is being processed.")
        date, line = 1, 1
        while date and line:
            date, line = file.readline(), file.readline()
            dates.append(date.strip())
            line = remove_stopwords(clean_text(line.strip()))
            news.append(line)

# Keep only the top 250 words for each day's news
news = keep_top_words(news, num_words=250)

# Save to CSV
data = {'Date': dates, 'news': news}
df = pd.DataFrame(data)
df.to_csv('news.csv', index=False)

#
# folder_path = "./shares"
# file_addresses = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
#
# dfs = []
#
# for filename in file_addresses:
#     df = pandas.read_csv(filename)
#     df.columns = [col + "_" + filename[-7:-4] if col != 'Date' else col for col in df.columns]
#     dfs.append(df)
#
# merged_df = dfs[0]
# for df in dfs[1:]:
#     merged_df = pandas.merge(merged_df, df, on='Date', how='outer')
#
# merged_df.sort_values(by='Date', inplace=True)
# merged_df.reset_index(drop=True, inplace=True)
#
# merged_df.to_csv('shares.csv', index=False)
