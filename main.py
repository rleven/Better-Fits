import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

import matplotlib.pyplot as plt
import scipy.stats as stats

class BFModels:
    def __init__(self):
        self.models = {}

    def add_model(self, name, model):
        self.models[name] = model

    def get_model(self, name):
        return self.models.get(name, None)

    def list_models(self):
        return list(self.models.keys())