from sentence_transformers import SentenceTransformer
import chromadb
import os
from flask import send_file, jsonify

class ImageHandler:
    def __init__(self):
        # Initialize ChromaDB client and collection
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("image_labels")
        
        # Initialize SentenceTransformer model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load labels from JSON file
        try:
            with open(os.path.join(os.path.dirname(__file__), '../labels.json'), 'r') as f:
                self.labels = json.load(f)
        except FileNotFoundError:
            self.labels = {}

    def populate_vector_database(self):
        for label, image_path in self.labels.items():
            existing_records = self.collection.get(where={"label": label})
            if len(existing_records["ids"]) == 0:
                embedding = self.embedder.encode(label).tolist()
                self.collection.add(
                    embeddings=[embedding],
                    documents=[label],
                    metadatas=[{"image_path": image_path}],
                    ids=[label]
                )

    def generate_image(self, query):
        try:
            # Compute embedding for the user query
            query_embedding = self.embedder.encode(query).tolist()
            
            # Search for the most similar label
            results = self.collection.query(query_embeddings=[query_embedding], n_results=1)
            
            if results["documents"]:
                matched_label = results["documents"][0][0]
                metadata = results["metadatas"][0][0]
                image_path = metadata["image_path"]
                
                if os.path.exists(image_path):
                    return send_file(image_path, mimetype='image/jpeg')
                else:
                    return jsonify({'error': 'Image not found'}), 404
            else:
                return jsonify({'error': 'No matching image found'}), 404

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500