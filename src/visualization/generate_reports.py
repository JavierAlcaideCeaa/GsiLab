import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def generate_sentiment_distribution_report(sentiment_data, output_dir):
    """
    Generates a bar chart showing the distribution of sentiments.

    Parameters:
    sentiment_data (pd.DataFrame): DataFrame containing sentiment counts.
    output_dir (str): Directory where the report will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_data['Sentiment'], y=sentiment_data['Count'], palette='viridis')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()

    report_path = os.path.join(output_dir, 'sentiment_distribution_report.png')
    plt.savefig(report_path)
    plt.close()
    print(f'Sentiment distribution report saved to {report_path}')

def generate_trend_report(trend_data, output_dir):
    """
    Generates a line chart showing sentiment trends over time.

    Parameters:
    trend_data (pd.DataFrame): DataFrame containing sentiment trends.
    output_dir (str): Directory where the report will be saved.
    """
    plt.figure(figsize=(12, 6))
    for sentiment in trend_data['Sentiment'].unique():
        subset = trend_data[trend_data['Sentiment'] == sentiment]
        plt.plot(subset['Date'], subset['Count'], marker='o', label=sentiment)

    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    report_path = os.path.join(output_dir, 'sentiment_trend_report.png')
    plt.savefig(report_path)
    plt.close()
    print(f'Sentiment trend report saved to {report_path}')