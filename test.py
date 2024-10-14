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

# Création ou récupération d'une collection dans Chroma
collection = client.get_or_create_collection("html_text_collection")

# 5. Traitement des fichiers HTML et XML et stockage des vecteurs dans Chroma
def process_html_and_xml_files(directory, file_type):
    files = [f for f in os.listdir(directory) if f.endswith(f'.{file_type}')]
    
    if not files:
        print(f"Aucun fichier {file_type} trouvé dans le répertoire {directory}.")
        return

    for file in files:
        file_path = os.path.join(directory, file)
        
        if file_type == "html":
            text = extract_text_from_html(file_path)
        else:
            text = extract_text_from_xml(file_path)
        
        vector = text_to_vector(text)
        collection.add(
            documents=[text],
            embeddings=[vector],
            ids=[file]
        )
        print(f"Fichier {file} traité et stocké dans Chroma.")
        print("\nVecteur complet :")
        print(vector)

# Fonction pour extraire le texte d'un fichier XML en utilisant lxml
def extract_text_from_xml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml")  # Utilisation de lxml pour le parsing XML
        # Extraction du texte en ignorant les balises
        return soup.get_text(separator=" ")

# Fonction pour effectuer une requête sur la base de données Chroma
def query_chroma(query_text, n_results=5):
    # Convertir le texte de la requête en vecteur
    query_vector = text_to_vector(query_text)
    
    # Effectuer la requête
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=['documents', 'distances']
    )
    
    return results

# Fonction pour obtenir les vecteurs et textes d'un document à partir de son ID
def get_vector_and_text_for_document(document_id):
    results = collection.get(
        ids=[document_id],
        include=['embeddings', 'documents']
    )
    if results['embeddings'] and results['documents']:
        return results['embeddings'][0], results['documents'][0]
    else:
        return None, None

# Fonction pour demander à l'utilisateur de choisir le type de fichier (HTML ou XML)
def choose_file_type():
    while True:
        choice = input("Quel type de fichier voulez-vous traiter ? (html/xml) : ").lower()
        if choice in ['html', 'xml']:
            return choice
        else:
            print("Veuillez entrer 'html' ou 'xml'.")

# Fonction pour demander à l'utilisateur le répertoire à utiliser
def choose_directory(file_type):
    if file_type == 'html':
        return "htmlDocument"  # Répertoire pour les fichiers HTML
    else:
        return "xmlDocument"  # Répertoire pour les fichiers XML

# Exemple d'utilisation : traiter tous les fichiers HTML et XML d'un répertoire
if __name__ == "__main__":
    file_type = choose_file_type()  # Demander à l'utilisateur s'il souhaite traiter HTML ou XML
    directory = choose_directory(file_type)  # Obtenir le répertoire en fonction du type de fichier
    process_html_and_xml_files(directory, file_type)

    # Exemple d'utilisation pour récupérer un vecteur et du texte pour un document spécifique
    document_id = input(f"Entrez le nom du fichier {file_type} que vous souhaitez récupérer (ex: 'index.html' ou 'document.xml') : ")
    vector, text = get_vector_and_text_for_document(document_id)

    if vector and text:
        print(f"Données pour {document_id}:")
        print("\nTexte extrait:")
        print(text)  # Affichage complet du texte
        print("\nVecteur complet :")
        print(vector)  # Affichage complet du vecteur
        
        # Statistiques simples sur le vecteur
        print("\nStatistiques du vecteur :")
        print("Minimum :", min(vector))
        print("Maximum :", max(vector))
        print("Moyenne :", sum(vector) / len(vector))
    else:
        print(f"Aucune donnée trouvée pour {document_id}")

    # Exemple d'utilisation de la fonction de requête
    query_text = input("Entrez le texte pour effectuer une requête sur Chroma : ")
    results = query_chroma(query_text)

    print(f"\nRésultats pour la requête : '{query_text}'")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"\nRésultat {i+1} (distance : {distance:.4f}) :")
        print(doc)
