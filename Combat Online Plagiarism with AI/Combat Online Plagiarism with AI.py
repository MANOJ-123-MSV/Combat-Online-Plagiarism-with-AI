import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')

def preprocess_document(doc):
    # Tokenize the document
    tokens = nltk.word_tokenize(doc)
    # Lowercase all words
    tokens = [word.lower() for word in tokens]
    return ' '.join(tokens)

def calculate_similarity(doc1, doc2):
    # Preprocess the documents
    doc1 = preprocess_document(doc1)
    doc2 = preprocess_document(doc2)
    
    # Vectorize the documents using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    
    # Calculate the cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    return similarity_matrix[0][1]

# Sample documents
doc1 = "This is a sample document to check for plagiarism."
doc2 = "This document is a sample used to check plagiarism."

similarity_score = calculate_similarity(doc1, doc2)
print(f"Similarity score: {similarity_score:.2f}")

# You can now use this function to compare any two documents
