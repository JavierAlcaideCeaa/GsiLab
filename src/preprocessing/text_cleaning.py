import re
import string
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y validación.
    """
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, val_data

def preprocess_sentiment140(file_path):
    """
    Carga y preprocesa el dataset Sentiment140.
    """
    # Definir las columnas del dataset
    columns = ["target", "id", "date", "flag", "user", "text"]
    data = pd.read_csv(file_path, encoding="latin-1", names=columns)

    # Mapear etiquetas
    data["sentiment"] = data["target"].map({0: "negative", 4: "positive"})

    # Limpiar los textos
    data["cleaned_text"] = data["text"].apply(clean_text)

    # Filtrar columnas relevantes
    data = data[["cleaned_text", "sentiment"]]

    return data

def clean_text(text):
    """
    Limpia el texto eliminando caracteres especiales, puntuación y stopwords.
    """
    # Eliminar caracteres especiales y convertir a minúsculas
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    
    # Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenizar el texto
    tokens = word_tokenize(text)
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def preprocess_data(texts, tokenizer, max_len=100, vocab_size=None):
    """
    Preprocesa los textos para convertirlos en tensores listos para el modelo.
    """
    # Limpieza de textos
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Tokenización
    sequences = [tokenizer(text) for text in cleaned_texts]
    
    # Padding y truncamiento
    padded_sequences = [seq[:max_len] + [0] * (max_len - len(seq)) for seq in sequences]
    tensor = torch.tensor(padded_sequences, dtype=torch.long)

    # Clamping de valores
    if vocab_size:
        tensor = torch.clamp(tensor, min=0, max=vocab_size - 1)
    return tensor