
import os
import faiss
import pickle
import sys

print("Starting debug script")
storage_folder = "storage"
index_path = os.path.join(storage_folder, "semantic.index")
docs_path = os.path.join(storage_folder, "docs.pkl")

if os.path.exists(index_path):
    print(f"Loading index from {index_path}...")
    try:
        index = faiss.read_index(index_path)
        print("Index loaded successfully.")
    except Exception as e:
        print(f"Failed to load index: {e}")
else:
    print("Index file not found.")

if os.path.exists(docs_path):
    print(f"Loading docs from {docs_path}...")
    try:
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        print(f"Docs loaded successfully. Count: {len(docs)}")
    except Exception as e:
        print(f"Failed to load docs: {e}")
else:
    print("Docs file not found.")

print("Debug script finished")
