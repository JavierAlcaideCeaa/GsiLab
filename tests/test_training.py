import unittest
from src.training.train_model import train_model, load_data

class TestTraining(unittest.TestCase):

    def test_load_data(self):
        # Test if data loading function works correctly
        data = load_data('data/processed/sample_data.csv')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_train_model(self):
        # Test if the model training function executes without errors
        try:
            model = train_model('data/processed/sample_data.csv')
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"train_model raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()