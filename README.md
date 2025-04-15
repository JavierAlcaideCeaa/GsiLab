# Sentiment Analysis on Social Media for Business Decision Making

This project implements a sentiment analysis system that leverages data mining and deep learning techniques to analyze social media posts. The primary goal is to classify sentiments as positive, negative, or neutral, providing businesses with insights to make informed marketing decisions.

## Project Structure

- **data/**
  - **raw/**: Contains raw data collected from social media APIs.
  - **processed/**: Stores processed data ready for analysis and modeling.
  - **reports/**: Holds the generated visual reports for business insights.

- **models/**
  - **rnn_model.py**: Defines the RNN model for sentiment analysis, including architecture and methods for training and prediction.
  - **lstm_model.py**: Defines the LSTM model for sentiment analysis, including architecture and methods for training and prediction.
  - **model_utils.py**: Contains utility functions for model handling, such as saving and loading models.

- **notebooks/**
  - **data_exploration.ipynb**: Jupyter notebook for exploratory data analysis, visualizing data distributions, and understanding sentiment trends.

- **src/**
  - **api/**
    - **twitter_api.py**: Functions to interact with the Twitter API, including authentication and data retrieval.
  - **preprocessing/**
    - **text_cleaning.py**: Functions for cleaning and preprocessing text data, such as removing stop words, punctuation, and applying tokenization.
  - **training/**
    - **train_model.py**: Contains the training logic for the sentiment analysis models, including data loading, model training, and saving the trained model.
  - **evaluation/**
    - **evaluate_model.py**: Functions to evaluate the performance of the trained models using metrics like accuracy, precision, and recall.
  - **visualization/**
    - **generate_reports.py**: Functions to generate visual reports from the analysis results, such as trend graphs and sentiment distribution charts.

- **tests/**
  - **test_api.py**: Unit tests for the Twitter API functions to ensure they work as expected.
  - **test_preprocessing.py**: Unit tests for the text cleaning functions to validate preprocessing steps.
  - **test_training.py**: Unit tests for the training functions to ensure models are trained correctly.
  - **test_evaluation.py**: Unit tests for the evaluation functions to validate model performance metrics.

- **requirements.txt**: Lists the dependencies required for the project, including libraries for data processing, machine learning, and visualization.

- **README.md**: Documentation for the project, including setup instructions, usage, and an overview of the system.

- **main.py**: Entry point for the application, coordinating data retrieval, model training, evaluation, and report generation.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd sentiment-analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Twitter API credentials in `src/api/twitter_api.py`.

4. Run the main application:
   ```
   python main.py
   ```

## Usage

- The system retrieves data from Twitter, processes it, trains sentiment analysis models, evaluates their performance, and generates visual reports.
- Explore the Jupyter notebook for data exploration and insights.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.