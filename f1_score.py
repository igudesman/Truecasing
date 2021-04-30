import numpy as np
import torch


class F1():
    def __init__(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted
      
    def _convert(self):
        actual_arr = np.array([])
        for company_name in self.actual:
            for char in company_name:
                if char.isupper():
                    actual_arr = np.append(actual_arr, 1)
                else:
                    actual_arr = np.append(actual_arr, 0)

        predicted_arr = np.array([])
        for company_name in self.predicted:
            for char in company_name:
                if char.isupper():
                    predicted_arr = np.append(predicted_arr, 1)
                else:
                    predicted_arr = np.append(predicted_arr, 0)
        
        actual_tensor, predicted_tensor = torch.tensor(actual_arr), torch.tensor(predicted_arr)
        return actual_tensor, predicted_tensor


    def f1_score(self):
        actual, predicted = self._convert()

        tp = (actual * predicted).sum().to(torch.float32)
        tn = ((1.0 - actual) * (1.0 - predicted)).sum().to(torch.float32)
        fp = ((1.0 - actual) * predicted).sum().to(torch.float32)
        fn = ((1.0 - predicted) * actual).sum().to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        return f1