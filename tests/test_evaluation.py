import unittest
from src.evaluation.evaluate_model import evaluate_model

class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        # Setup code to initialize variables or load models if needed
        self.y_true = [1, 0, 1, 1, 0]  # Example true labels
        self.y_pred = [1, 0, 1, 0, 0]  # Example predicted labels

    def test_accuracy(self):
        accuracy = evaluate_model(self.y_true, self.y_pred, metric='accuracy')
        self.assertEqual(accuracy, 0.8)  # Example expected accuracy

    def test_precision(self):
        precision = evaluate_model(self.y_true, self.y_pred, metric='precision')
        self.assertEqual(precision, 1.0)  # Example expected precision

    def test_recall(self):
        recall = evaluate_model(self.y_true, self.y_pred, metric='recall')
        self.assertEqual(recall, 0.75)  # Example expected recall

    def test_f1_score(self):
        f1_score = evaluate_model(self.y_true, self.y_pred, metric='f1_score')
        self.assertEqual(f1_score, 0.8571428571428571)  # Example expected F1 score

if __name__ == '__main__':
    unittest.main()