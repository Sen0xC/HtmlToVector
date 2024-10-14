import os
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# 1. Fonction pour extraire le texte d'un fichier HTML
def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        # Extraction du texte en ignorant les balises de script, style, etc.
        for script in soup(["script", "style"]):
            script.extract()
        return soup.get_text(separator=" ")

# 2. Chargement du modèle d'encodage de texte
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Fonction pour transformer du texte en vecteur
def text_to_vector(text):
    return model.encode(text).tolist()

# 4. Initialisation de Chroma
client = chromadb.Client()

# Création d'une collection dans Chroma
collection = client.create_collection("html_text_collection")

# 5. Traitement des fichiers HTML et stockage des vecteurs dans Chroma
def process_html_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            file_path = os.path.join(directory, filename)
            # Extraire le texte du fichier HTML
            text = extract_text_from_html(file_path)
            # Convertir le texte en vecteur
            vector = text_to_vector(text)
            # Ajouter le texte et le vecteur à la collection Chroma
            collection.add(
                documents=[text],
                embeddings=[vector],
                ids=[filename]
            )
            print(f"Fichier {filename} traité et stocké dans Chroma.")

# Exemple d'utilisation : traiter tous les fichiers HTML d'un répertoire
process_html_files("htmlDocument")

def get_vector_and_text_for_document(document_id):
    results = collection.get(
        ids=[document_id],
        include=['embeddings', 'documents']
    )
    if results['embeddings'] and results['documents']:
        return results['embeddings'][0], results['documents'][0]
    else:
        return None, None

# Exemple d'utilisation
document_id = "index.html"  # Remplacez par le nom de votre fichier HTML
vector, text = get_vector_and_text_for_document(document_id)

if vector and text:
    print(f"Données pour {document_id}:")
    print("\nTexte extrait:")
    print(text[:500] + "..." if len(text) > 500 else text)  # Affiche les 500 premiers caractères du texte
    
    print("\nVecteur:")
    print("Dimensions du vecteur:", len(vector))
    print("Contenu du vecteur:")
    print(vector)
    
    # Statistiques simples sur le vecteur
    print("\nStatistiques du vecteur:")
    print("Minimum:", min(vector))
    print("Maximum:", max(vector))
    print("Moyenne:", sum(vector) / len(vector))
else:
    print(f"Aucune donnée trouvée pour {document_id}")
