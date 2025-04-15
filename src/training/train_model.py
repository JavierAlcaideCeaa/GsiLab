import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.rnn_model import RNNModel
from models.lstm_model import LSTMModel
from src.preprocessing.text_cleaning import preprocess_data
from src.preprocessing.text_cleaning import preprocess_sentiment140

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data['text'], data['sentiment']

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)



import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

def train_and_save_model(model_type='rnn', train_file=None, val_file=None, tokenizer=None):
    import pandas as pd
    import torch.nn as nn
    import torch.optim as optim

    # Verifica que los archivos de entrenamiento y validación sean proporcionados
    if train_file is None or val_file is None:
        raise ValueError("Both train_file and val_file must be provided.")

    # Verifica que el tokenizador no sea None
    if tokenizer is None:
        raise ValueError("A tokenizer must be provided to preprocess the data.")

    # Cargar datos procesados
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)

    # Eliminar filas con valores nulos
    train_data = train_data.dropna(subset=["cleaned_text"])
    val_data = val_data.dropna(subset=["cleaned_text"])

    # Tokenizar los textos
    train_texts = [torch.tensor(tokenizer(text), dtype=torch.long) for text in train_data["cleaned_text"]]
    val_texts = [torch.tensor(tokenizer(text), dtype=torch.long) for text in val_data["cleaned_text"]]

    # Rellenar o recortar las secuencias a una longitud fija
    max_length = 85  # Longitud fija (puedes ajustarla según tus datos)
    train_texts = pad_sequence(train_texts, batch_first=True, padding_value=0)[:, :max_length]
    val_texts = pad_sequence(val_texts, batch_first=True, padding_value=0)[:, :max_length]

    # Convertir etiquetas a índices
    train_labels = pd.factorize(train_data["sentiment"])[0]
    val_labels = pd.factorize(val_data["sentiment"])[0]

    # Crear DataLoaders
    train_dataset = TensorDataset(train_texts, torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(val_texts, torch.tensor(val_labels, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Calcular vocab_size
    vocab_size = max([token.max().item() for token in train_texts]) + 1
    num_classes = len(set(train_labels))

    # Crear modelo
    if model_type == 'rnn':
        model = RNNModel(input_size=vocab_size, hidden_size=128, output_size=num_classes)
    elif model_type == 'lstm':
        model = LSTMModel(input_size=vocab_size, hidden_size=128, output_size=num_classes)
    else:
        raise ValueError("Invalid model type. Choose 'rnn' or 'lstm'.")

    initialize_weights(model)

    # Configurar optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Entrenamiento
    for epoch in range(10):  # Reduce el número de épocas para pruebas rápidas
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            print(f"Training Loss: {loss.item()}")

        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")

    # Guardar modelo
    torch.save(model.state_dict(), f"models/{model_type}_model.pth")