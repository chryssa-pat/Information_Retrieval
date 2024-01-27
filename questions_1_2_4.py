import os
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import ast
import re
from matplotlib import pyplot as plt
import csv

# Question 1

# defaultdict to store the inverted index
inverted_index = defaultdict(list)

# Dictionary to store the  documents each word appears in
document_count = {}

path = (r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\docs")
os.chdir(path)


for file in os.listdir(path):
    file_path = os.path.join(path, file)
    with open(file_path, 'r') as folder:
        text = folder.read()

        dictionary = text.split()

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
inverted_index_csv = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\inverted_index.csv"

# Print the inverted index
for word, documents in inverted_index.items():
    print(f"{word}: {documents}")

# Save the inverted index into a CSV file
with open(inverted_index_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Word', 'Documents'])

    for word, documents in inverted_index.items():
        csv_writer.writerow([word, documents])

# Question 2
# Calculate IDF for each word and store it in a dictionary
idf = {}
for word, documents in document_count.items():
    idf_value = math.log10((len([f for f in os.listdir(path)])) / len(documents))
    idf[word] = idf_value

# Create a DataFrame with IDF values
idf = pd.DataFrame.from_dict(idf, orient='index', columns=['IDF'])

csv_file_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\idf.csv"
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

csv_file_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\doc_tf.csv"
# Save the DataFrame to a CSV file
tf.to_csv(csv_file_path, index=False)

# Iterate through the TF and IDF DataFrames to calculate TF-IDF
tfidf = tf.copy()
# Multiply each TF value by the corresponding IDF value
for column in tfidf.columns:
    tfidf[column] = tfidf[column] * idf.loc[column, 'IDF']

csv_file_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\doc_weights.csv"
# Save the DataFrame to a CSV file
tfidf.to_csv(csv_file_path, index=False)

query_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\Queries_20"
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
    word_counts = {word: 0 for word in common_words}

    for word in common_words:
        # Count the number of occurrences of the word in the question
        pattern = r'\b' + re.escape(word) + r'\b'
        count = len(re.findall(pattern, question.upper()))
        # Update the DataFrame with the log-transformed count
        if count > 0:
            df.at[i, word] = 1 + math.log10(count)

df = df.fillna(0)
csv_file_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\query_tf.csv"
# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

query_tfidf = df.multiply(idf['IDF'], axis=1)
csv_file_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\query_weights.csv"
# Save the DataFrame to a CSV file
query_tfidf.to_csv(csv_file_path, index=False)

# Calculate the Euclidean norm for each row of doc_weights
euclidean_norms = np.linalg.norm(tfidf.values, axis=1)
# Create a DataFrame with the Euclidean norm values
euclidean_norms_d = pd.DataFrame({'Euclidean Norm': euclidean_norms}, index=tfidf.index)

# Calculate the Euclidean norm for each row of query_weights
euclidean_norms = np.linalg.norm(query_tfidf.values, axis=1)
# Create a DataFrame with the Euclidean norm values
euclidean_norms_q = pd.DataFrame({'Euclidean Norm': euclidean_norms}, index=query_tfidf.index)

# Calculate the dot product
dot_products = np.dot(tfidf.values, query_tfidf.values.T)
# Create a DataFrame with the dot product values
dot_product = pd.DataFrame(dot_products, index=tfidf.index, columns=query_tfidf.index)

# Calculate the product of the Euclidean norms
norm_product = euclidean_norms_d.values @ euclidean_norms_q.values.T
# Create a DataFrame with the result
norm_product = pd.DataFrame(norm_product, index=euclidean_norms_d.index, columns=euclidean_norms_q.index)

# Calculate the cosine similarity
cosine_similarity = dot_product.div(norm_product)
print(cosine_similarity)

# Question 4
# Precision - Recall and MAP Metrics
#cosine similarity
# Sort each column and replace values with doc IDs
sorted_doc_ids = cosine_similarity.apply(lambda col: col.sort_values(ascending=False).index)
csv_file_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\cosine_similarity.csv"
# Save the DataFrame to a CSV file
sorted_doc_ids.to_csv(csv_file_path, index=False)
# Convert document IDs to integers and get the retrieved docs
retrieved_docs = {query_id: [int(doc_id) for doc_id in doc_ids] for query_id, doc_ids in sorted_doc_ids.items()}

#colbert
colbert_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\colbert_result.csv"
query_docs = {}
# Read the ColBERT results into a DataFrame
with open(colbert_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header row

    for query_id, row in enumerate(csvreader):
        # Parse the string representation of the list into an actual list
        doc_ids = ast.literal_eval(row[1])

        # Convert each element in the list to an integer
        doc_ids = [int(id) for id in doc_ids]

        # Ensure doc_ids is a list of integers
        if isinstance(doc_ids, list) and all(isinstance(id, int) for id in doc_ids):
            query_docs[query_id] = doc_ids
# Get the relevant docs
file_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\Relevant_20"
relevant_docs = {}
with open(file_path, 'r') as file:
    for query_id, line in enumerate(file):
        # Convert all space-separated numbers in the line to integers and store them as a set
        doc_ids = set(map(int, line.split()))
        # Assign this set to the corresponding query ID
        relevant_docs[query_id] = doc_ids

def calculate_metrics(relevant_docs, retrieved_docs):
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
    avg_precision = sum(precisions) / len(relevant_docs)

    return precisions, recalls, avg_precision

def calculate_area(recalls, precisions):
    area = 0.0
    for i in range(1, len(recalls)):
        # Calculate the area of the trapezoid
        width = recalls[i] - recalls[i-1]
        height = (precisions[i] + precisions[i-1]) / 2
        area += width * height
    return area


def precision_at_k(relevant_docs, retrieved_docs, k):
    if len(retrieved_docs) > k:
        retrieved_docs = retrieved_docs[:k]
    relevant_count = sum([1 for doc in retrieved_docs if doc in relevant_docs])
    return relevant_count / k


queries = 20
top_k = 400
avg_precisions = []
avg_precisions_colbert = []
areas_cosine_similarity = []
areas_colbert = []
precisions_at_k_colbert = []
precisions_at_k_cosine = []

for query_id, relevant_docs_set in relevant_docs.items():
    retrieved_docs_list = retrieved_docs.get(query_id, [])
    precisions, recalls, avg_precision = calculate_metrics(relevant_docs_set, retrieved_docs_list)
    avg_precisions.append(avg_precision)
    map_values = sum(avg_precisions) / queries


    # Calculate precision@k for Cosine Similarity
    precision_at_k_cosine = precision_at_k(relevant_docs_set, retrieved_docs_list, top_k)
    precisions_at_k_cosine.append(precision_at_k_cosine)

    # Print or store these values for comparison
    print(f"Precision@{top_k} for Cosine Similarity (Query {query_id + 1}): {precision_at_k_cosine}")

    plt.figure()
    plt.plot(recalls, precisions, marker='o', label = 'Cosine Similarity')
    plt.title(f"Precision - Recall for Query {query_id + 1}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.show()

    # After calculating precisions and recalls for each query
    area_cosine_similarity = calculate_area(recalls, precisions)
    areas_cosine_similarity.append(area_cosine_similarity)

    # Now you can print or plot the AUC valuesD
    print(f"Area for Cosine Similarity (Query {query_id + 1}): {area_cosine_similarity}")

for query_id, relevant_docs_set in relevant_docs.items():
    retrieved_docs_list_colbert = query_docs.get(query_id, [])
    precisions_colbert, recalls_colbert, avg_precision_colbert = calculate_metrics(relevant_docs_set, retrieved_docs_list_colbert)
    avg_precisions_colbert.append(avg_precision_colbert)
    map_values_colbert = sum(avg_precisions_colbert) / queries

    # Calculate precision@k for ColBERT
    precision_at_k_colbert = precision_at_k(relevant_docs_set, retrieved_docs_list_colbert, top_k)
    precisions_at_k_colbert.append(precision_at_k_colbert)

    # Print or store these values for comparison
    print(f"Precision@{top_k} for ColBERT (Query {query_id + 1}): {precision_at_k_colbert}")

    plt.figure()
    plt.plot(recalls_colbert, precisions_colbert, marker='o', label = 'ColBERT', color = 'orange')
    plt.title(f"Precision - Recall for Query {query_id + 1}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.show()

    # After calculating precisions and recalls for each query
    area_colbert = calculate_area(recalls_colbert, precisions_colbert)
    areas_colbert.append(area_colbert)

    # Now you can print or plot the AUC values
    print(f"Area for ColBERT (Query {query_id + 1}): {area_colbert}")


print(f"MAP Metric for Cosine Similarity: {map_values}")
print(f"MAP Metric for ColBERT: {map_values_colbert}")

# Calculate the mean area to compare the general efficiency of both models
mean_area_cosine_similarity = sum(areas_cosine_similarity) / queries
mean_area_colbert = sum(areas_colbert) / queries

print(f"Mean Area under Precision-Recall Curve for Cosine Similarity: {mean_area_cosine_similarity}")
print(f"Mean Area under Precision-Recall Curve for ColBERT: {mean_area_colbert}")

mean_precision_at_k_cosine_similarity = sum(precisions_at_k_cosine)/queries
mean_precision_at_k_colbert = sum(precisions_at_k_colbert)/queries

print(f"Mean Value of Precision@{top_k} for Cosine Similarity: {mean_precision_at_k_cosine_similarity}")
print(f"Mean Value of Precision@{top_k} for ColBERT: {mean_precision_at_k_colbert}")
