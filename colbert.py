import os
from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv

# Replace these paths with your actual file paths
colbert_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\colbert_result.csv"
relevant_documents_path = r"C:\Users\chryssa_pat\PycharmProjects\pythonProject\Relevant_20"

# Read the ColBERT results into a DataFrame
colbert = pd.read_csv(colbert_path)

# Read relevant documents from file
relevant_documents_dict = {}
with open(relevant_documents_path, 'r') as rel_file:
    for query_id, line in enumerate(rel_file):
        rel_docs = [int(doc) for doc in line.strip().split()]
        relevant_documents_dict[query_id] = rel_docs

# Initialize a list to store precision and recall values for each query
precision_recall_list = []
# Dictionary to store lists of DocumentIDs for each query
query_document_ids = defaultdict(list)

# Create an empty list to store data for DataFrame
data = []

# Read your CSV file
with open(colbert_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    # Iterate over each row
    for row in reader:
        # Extract query and DocumentIDs
        query = row['Query']
        document_ids = eval(row['DocumentIDs'])  # Convert string representation to list

        # Save DocumentIDs in the corresponding list for the query
        query_document_ids[query].extend(document_ids)

        # Append data to the list
        data.append({'Query': query, 'DocumentIDs': document_ids})

query_document_df = pd.DataFrame(data)


# Display the DataFrame
print(query_document_df)
# Iterate through queries
for query_id, row in query_document_df.iterrows():
    query_document_df = row['DocumentIDs']

    # Retrieve relevant documents for the current query
    relevant_docs = relevant_documents_dict.get(query_id, [])
    print(relevant_docs)
    # Compute relevance function X_Q (indexing starts with zero)
    X_Q = np.zeros(len(query_document_df), dtype=bool)
    for i, doc_id in enumerate(query_document_df):
        if doc_id in relevant_docs:
            X_Q[i] = True

    # Compute precision and recall values (indexing starts with zero)
    M = len(relevant_docs)
    P_Q = np.cumsum(X_Q) / np.arange(1, len(query_document_df) + 1)
    R_Q = np.cumsum(X_Q) / M

    precision_recall_list.append((P_Q, R_Q, query_document_df, relevant_docs))

# Plot precision-recall curve for each query
for query_id, (precision_values, recall_values, sorted_doc_ids, relevant_docs) in enumerate(precision_recall_list,
                                                                                            start=1):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Plot dots only for documents that are both in sorted and relevant
    common_docs = set(sorted_doc_ids) & set(relevant_docs)
    common_indices = [sorted_doc_ids.index(doc) for doc in common_docs]

    plt.scatter(np.array(recall_values)[common_indices], np.array(precision_values)[common_indices],
                marker='o', color='k', label='Common Documents')

    plt.title(f'Precision-Recall Curve - Query {query_id}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.xticks(np.arange(0, 1.1, 0.2))

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
