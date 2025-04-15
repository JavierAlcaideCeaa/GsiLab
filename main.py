from src.preprocessing.text_cleaning import preprocess_sentiment140, split_data
from src.training.train_model import train_and_save_model

def main():
    # Ruta al archivo Sentiment140
    file_path = "training.1600000.processed.noemoticon.csv"

    # Preprocesar el dataset
    data = preprocess_sentiment140(file_path)

    # Dividir los datos en entrenamiento y validación
    train_data, val_data = split_data(data)

    # Reducir el tamaño del dataset para pruebas
    train_data = train_data.sample(10000, random_state=42)  # 10,000 filas para entrenamiento
    val_data = val_data.sample(2000, random_state=42)       # 2,000 filas para validación

    # Guardar los datos procesados
    train_data.to_csv("data/processed/sentiment140_train.csv", index=False)
    val_data.to_csv("data/processed/sentiment140_val.csv", index=False)

    # Entrenar el modelo
    tokenizer = lambda x: [ord(c) for c in x if isinstance(x, str)]  # Tokenizador basado en caracteres
    train_and_save_model(
        model_type='lstm',
        train_file="data/processed/sentiment140_train.csv",
        val_file="data/processed/sentiment140_val.csv",
        tokenizer=tokenizer
    )

if __name__ == "__main__":
    main()