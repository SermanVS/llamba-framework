import pandas as pd
import torch
from torch import nn
import shap
import numpy as np

class BioAgeModel():
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.model.freeze()

    def inference(self, data: pd.DataFrame, device: torch.device):
        self.model.to(device)
        self.model.eval()
        if str(device) == "cuda":
            res = self.model(torch.from_numpy(data.values)).cuda().detach().numpy().ravel()
        else:
            res = self.model(torch.from_numpy(data.values)).cpu().detach().numpy().ravel()
        return res

    def get_top_shap(self, n, data, feats, shap_dict):
        top_shap = {}
        np.random.seed(0)
        torch.manual_seed(0)
        explainer = shap_dict['explainer']
        shap_values_trgt = explainer.shap_values(data.loc[0, feats].values)
        base_value = explainer.expected_value[0]

        explanation = shap.Explanation(
            values=shap_values_trgt,
            base_values=base_value,
            data=data.loc[0, feats].values,
            feature_names=feats)

        permutation = np.array(explanation.values).argsort()
        
        # Top-n values
        top_shap['values'] = np.array(explanation.values)[permutation][-n:].tolist()
        top_shap['data'] = np.array(explanation.data)[permutation][-n:].tolist()
        top_shap['feats'] = np.array(feats)[permutation][-n:].tolist()
        top_shap['explanation'] = explanation
        top_shap['explainer'] = shap_dict['explainer']
        return top_shap