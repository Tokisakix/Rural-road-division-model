import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LabelValidator:
    def __init__(self):
        return

    def run(self, ground_truth, predictions):
        '''
        Initializes the CAMValidator with ground truth and predicted labels, each of shape (1024, 1024, 3).
        '''
        self.ground_truth = ground_truth
        self.predictions = predictions

    def flatten_labels(self, labels):
        '''
        Flatten the 3D label tensor (1024, 1024, 3) to a 1D array for evaluation.
        '''
        return labels.view(-1).cpu().numpy()

    def calculate_accuracy(self):
        '''
        Calculate and return the accuracy of the predictions.
        '''
        gt_flat = self.flatten_labels(self.ground_truth)
        pred_flat = self.flatten_labels(self.predictions)
        return accuracy_score(gt_flat, pred_flat)

    def calculate_precision(self):
        '''
        Calculate and return the precision of the predictions.
        '''
        gt_flat = self.flatten_labels(self.ground_truth)
        pred_flat = self.flatten_labels(self.predictions)
        return precision_score(gt_flat, pred_flat)

    def calculate_recall(self):
        '''
        Calculate and return the recall of the predictions.
        '''
        gt_flat = self.flatten_labels(self.ground_truth)
        pred_flat = self.flatten_labels(self.predictions)
        return recall_score(gt_flat, pred_flat)

    def calculate_f1_score(self):
        '''
        Calculate and return the F1 score of the predictions.
        '''
        gt_flat = self.flatten_labels(self.ground_truth)
        pred_flat = self.flatten_labels(self.predictions)
        return f1_score(gt_flat, pred_flat)

# Example usage:
# validator = LabelValidator(ground_truth_labels, predicted_labels)
# accuracy = validator.calculate_accuracy()
# precision = validator.calculate_precision()
# recall = validator.calculate_recall()
# f1 = validator.calculate_f1_score()

def get_PartF():
    partf = LabelValidator()
    return partf