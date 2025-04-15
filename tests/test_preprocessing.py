import unittest
from src.preprocessing.text_cleaning import clean_text

class TestTextCleaning(unittest.TestCase):

    def test_clean_text_removes_punctuation(self):
        text = "Hello, world! This is a test."
        expected = "Hello world This is a test"
        result = clean_text(text)
        self.assertEqual(result, expected)

    def test_clean_text_removes_stop_words(self):
        text = "This is a sample text with some stop words."
        expected = "sample text stop words"
        result = clean_text(text)
        self.assertEqual(result, expected)

    def test_clean_text_tokenization(self):
        text = "Tokenize this sentence."
        expected = ["Tokenize", "this", "sentence"]
        result = clean_text(text, tokenize=True)
        self.assertEqual(result, expected)

    def test_clean_text_empty_string(self):
        text = ""
        expected = ""
        result = clean_text(text)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()