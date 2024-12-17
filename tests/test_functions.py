from llamba_library.bioage_model import BioAgeModel
import numpy as np
import pandas as pd
import torch
import unittest

class TestAnalyzeFunction(unittest.TestCase):
    def test_query(self):        
        # Prepare data to analyze
        np.random.seed(0)
        torch.manual_seed(0)
        num_features = 10
        features = np.random.randint(low=1, high=150, size=num_features).astype(np.float32)
        age =  np.random.randint(low=10, high=90)

        data = pd.DataFrame([{f'Feature_{i}' : features[i] for i in range(num_features)}])

        # Prepare a BioAge model
        class DummyBioAgeModel(torch.nn.Module): 
            def __init__(self): 
                super(DummyBioAgeModel, self).__init__()
                self.linear = torch.nn.Linear(10, 1)

            def forward(self, x):
                x = self.linear(x)
                return abs(x)
            
            def freeze(self): pass
            
        model = DummyBioAgeModel()
        bioage_model = BioAgeModel(model)
        res = bioage_model.inference(data, torch.device('cpu'))
        self.assertIsInstance(res[0], np.float32)
            
if __name__ == '__main__':
    unittest.main()