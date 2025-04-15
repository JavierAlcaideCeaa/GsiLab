import unittest
from src.api.twitter_api import fetch_tweets, authenticate_api

class TestTwitterAPI(unittest.TestCase):

    def setUp(self):
        self.api_key = 'your_api_key'
        self.api_secret_key = 'your_api_secret_key'
        self.access_token = 'your_access_token'
        self.access_token_secret = 'your_access_token_secret'

    def test_authenticate_api(self):
        auth = authenticate_api(self.api_key, self.api_secret_key, self.access_token, self.access_token_secret)
        self.assertIsNotNone(auth)

    def test_fetch_tweets(self):
        query = 'sentiment analysis'
        tweets = fetch_tweets(query, self.api_key, self.api_secret_key, self.access_token, self.access_token_secret)
        self.assertIsInstance(tweets, list)
        self.assertGreater(len(tweets), 0)

if __name__ == '__main__':
    unittest.main()