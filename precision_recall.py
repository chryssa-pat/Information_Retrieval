import os
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import ast
import re
from matplotlib import pyplot as plt
import csv

inverted_index = defaultdict(list)

path = (r"C:\Users\me\PycharmProjects\pythonProject1\docs")
os.chdir(path)
words_set = set()

document_count = {}  # Dictionary to store the count of documents each word appears in

for file in os.listdir(path):
    file_path = os.path.join(path, file)
    with open(file_path, 'r') as folder:
        text = folder.read()
        dictionary = text.split()
        words_set = words_set.union(set(dictionary))
        # Create a dictionary to store word counts for each document
        count = {}

        for word in dictionary:
            count[word] = count.get(word, 0) + 1

        # Update the inverted index with word counts for the current document
        for word, count in count.items():
            inverted_index[word].append((file, count))

            # Update the document count for the current word
            if word in document_count:
                document_count[word].add(file)
            else:
                document_count[word] = {file}

print(inverted_index)

# Calculate IDF for each word and store it in a dictionary
idf = {}
for word, documents in document_count.items():
    idf_value = math.log10((len([f for f in os.listdir(path)])) / len(documents))
    idf[word] = idf_value
# Create a DataFrame with IDF values
idf = pd.DataFrame.from_dict(idf, orient='index', columns=['IDF'])
print(idf)
csv_file_path = r"C:\Users\me\PycharmProjects\pythonProject1\idf.csv"
# Save the DataFrame to a CSV file
idf.to_csv(csv_file_path, index=False)

# Create a dictionary to store TF values for each word in each document
tf = {}
for word, occurrences in inverted_index.items():
    for document, count in occurrences:
        if document in tf:
            tf[document][word] = 1 + math.log10(count)
        else:
            tf[document] = {word: 1 + math.log10(count)}

# Create a DataFrame where rows are document titles and columns are words with their TF values
tf = pd.DataFrame.from_dict(tf, orient='index')
tf.fillna(0, inplace=True)  # Fill missing values with 0

# Now you have a DataFrame with TF values calculated as 1 + log(term_freq) for each word in each document
csv_file_path = r"C:\Users\me\PycharmProjects\pythonProject1\doc_tf.csv"
# Save the DataFrame to a CSV file
tf.to_csv(csv_file_path, index=False)
print(tf)

# Iterate through the TF and IDF DataFrames to calculate TF-IDF
tfidf = tf.copy()  # Create a copy of the TF DataFrame

# Multiply each TF value by the corresponding IDF value
for column in tfidf.columns:
    tfidf[column] = tfidf[column] * idf.loc[column, 'IDF']

# Now, tfidf_df contains the TF-IDF values
print(tfidf)

csv_file_path = r"C:\Users\me\PycharmProjects\pythonProject\doc_weights.csv"

# Save the DataFrame to a CSV file
tfidf.to_csv(csv_file_path, index=False)

query_path = r"C:\Users\me\PycharmProjects\pythonProject1\Queries_20"

# Read questions from the file
with open(query_path, 'r') as file:
    questions = file.readlines()

# Create a list of words from the inverted index
all_words = list(inverted_index.keys())

# Create a DataFrame with zeros
df = pd.DataFrame(0, index=range(len(questions)), columns=all_words)

# Iterate through questions and update the DataFrame
count = 0
for i, question in enumerate(questions):
    words = question.upper().split()
    # Find common words between the question and the inverted index
    common_words = set(words) & set(inverted_index)
    print(words)
    word_counts = {word: 0 for word in common_words}

    for word in common_words:
        # Count the number of occurrences of the word in the question
        pattern = r'\b' + re.escape(word) + r'\b'
        count = len(re.findall(pattern, question.upper()))

        # Update the DataFrame with the log-transformed count
        if count > 0:
            df.at[i, word] = 1 + math.log10(count)

df = df.fillna(0)

# Now, df is the DataFrame where each row represents a question, and columns represent words with values 1 or 0
csv_file_path = r"C:\Users\me\PycharmProjects\pythonProject1\query_tf.csv"
# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)
print(df)

query_tfidf = df.multiply(idf['IDF'], axis=1)
print(query_tfidf)

csv_file_path = r"C:\Users\me\PycharmProjects\pythonProject1\query_weights.csv"

# Save the DataFrame to a CSV file
query_tfidf.to_csv(csv_file_path, index=False)

# Calculate the Euclidean norm for each row of doc_weights
euclidean_norms = np.linalg.norm(tfidf.values, axis=1)

# Create a DataFrame with the Euclidean norm values
euclidean_norms_d = pd.DataFrame({'Euclidean Norm': euclidean_norms}, index=tfidf.index)

# Now, euclidean_norms_df contains the Euclidean norm values for each row in tfidf_df
print(euclidean_norms_d)

# Calculate the Euclidean norm for each row of query_weights
euclidean_norms = np.linalg.norm(query_tfidf.values, axis=1)

# Create a DataFrame with the Euclidean norm values
euclidean_norms_q = pd.DataFrame({'Euclidean Norm': euclidean_norms}, index=query_tfidf.index)

# Now, euclidean_norms_idf_df contains the Euclidean norm values for each row in idf_multiplied_df
print(euclidean_norms_q)

dot_products = np.dot(tfidf.values, query_tfidf.values.T)

# Create a DataFrame with the dot product values
dot_product = pd.DataFrame(dot_products, index=tfidf.index, columns=query_tfidf.index)

# Now, dot_product_df contains the dot product values for each pair of rows between idf_multiplied_df and tfidf_df
print(dot_product)

result_df = euclidean_norms_d.values @ euclidean_norms_q.values.T

# Create a DataFrame with the result
result_df = pd.DataFrame(result_df, index=euclidean_norms_d.index, columns=euclidean_norms_q.index)

# Now, result_df contains the result of multiplying euclidean_norms_idf_df with the transpose of euclidean_norms_df
print(result_df)

cosine_similarity = dot_product.div(result_df)

# Now, division_result_df contains the element-wise division results
print(cosine_similarity)
# Sort each column (originally rows/docs) and replace values with index (doc IDs)
sorted_doc_ids = cosine_similarity.apply(lambda col: col.sort_values(ascending=False).index)

csv_file_path = r"C:\Users\me\PycharmProjects\pythonProject1\cosine_similarity.csv"
# Save the DataFrame to a CSV file
sorted_doc_ids.to_csv(csv_file_path, index=False)
# Convert document IDs in retrieved to integers
retrieved_docs = {query_id: [int(doc_id) for doc_id in doc_ids] for query_id, doc_ids in sorted_doc_ids.items()}

file_path = r"C:\Users\me\PycharmProjects\pythonProject1\Relevant_20"
relevant_docs = {}
with open(file_path, 'r') as file:
    for query_id, line in enumerate(file):
        # Convert all space-separated numbers in the line to integers and store them as a set
        doc_ids = set(map(int, line.split()))
        # Assign this set to the corresponding query ID
        relevant_docs[query_id] = doc_ids

def calculate_precision_recall(relevant_docs, retrieved_docs):
    ret_count = 0
    rel_count = 0
    precisions = []
    recalls = []

    # Iterate through each relevant document
    for doc in retrieved_docs:
        ret_count += 1
        # Check if the relevant document is in the retrieved documents
        if doc in relevant_docs:
            rel_count += 1
            precision = rel_count / ret_count
            precisions.append(precision)
            recall = rel_count / len(relevant_docs)
            recalls.append(recall)

    return precisions, recalls

colbert_path = r"C:\Users\me\PycharmProjects\pythonProject1\colbert_result.csv"
# Read the ColBERT results into a DataFrame
query_docs = {}

with open(colbert_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header row

    for query_id, row in enumerate(csvreader):
        # Parse the string representation of the list into an actual list
        doc_ids = ast.literal_eval(row[1])

        # Ensure doc_ids is a list and contains integers
        if isinstance(doc_ids, list) and all(isinstance(id, int) for id in doc_ids):
            query_docs[query_id] = doc_ids

for query_id, relevant_docs_set in relevant_docs.items():
    retrieved_docs_list = retrieved_docs.get(query_id, [])
    retrieved_docs_list_colbert = query_docs.get(query_id, [])
    precisions, recalls = calculate_precision_recall(relevant_docs_set, retrieved_docs_list)
    precisions_colbert, recalls_colbert = calculate_precision_recall(relevant_docs_set, retrieved_docs_list_colbert)
    plt.figure()
    plt.plot(recalls, precisions, marker='o', label = 'Cosine Similarity')
    plt.plot(recalls_colbert, precisions_colbert, marker = 'o', label = 'ColBERT')
    plt.title(f"Precision - Recall for Query {query_id + 1}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.show()
