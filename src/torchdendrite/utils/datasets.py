import torch
from torch.utils.data import Dataset
from sklearn import datasets
import numpy as np
import math

class SimulatedDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx]).unsqueeze(0).type(torch.FloatTensor)
        y = torch.tensor(self.y[idx]).unsqueeze(0).type(torch.FloatTensor)
        return x, y

class SimulatedRegression:
    
    @staticmethod
    def linear(start=0, end=10, slope=2, intercept=3, err_std=1, n_samples=1000):
        X = np.linspace(start, end, n_samples)
        y = intercept + slope*X
        y = y + np.random.normal(loc=0.0, scale=err_std, size=y.shape)
        dataset = SimulatedDataset(X,y)
        return dataset

    @staticmethod
    def simple_sinusoid(start=0, num_periods=1, frequency=1, amplitude=1, phase_shift=0, err_std=0.1, n_samples=1000):
        X = np.linspace(start, num_periods*2*math.pi, n_samples)
        y = amplitude * np.sin(frequency*X + phase_shift)
        y = y + np.random.normal(loc=0.0, scale=err_std, size=y.shape)
        dataset = SimulatedDataset(X,y)
        return dataset
        
    @staticmethod
    def additive_sinusoid(start=0, num_periods=1, frequencies=(1,2), amplitudes=(1,1), phase_shifts=(0,0), n_samples=1000):

        assert len(frequencies) == len(amplitudes) == len(phase_shifts), "Ensure same length tuples of frequencies, amplitudes and phaseshifts are supplied"
        X = np.linspace(start, num_periods*2*math.pi, n_samples)
        y = np.zeros_like(X)
        
        for freq, amp, ps in zip(frequencies, amplitudes, phase_shifts):
            y += amp * np.sin(freq*X + ps)

        dataset = SimulatedDataset(X,y)
        return dataset

    @staticmethod
    def damped_sinusoid(start=0, num_periods=1, decay=0.1, frequency=1, amplitude=1, phase_shift=0, err_std=0.1, n_samples=1000):
        X = np.linspace(start, num_periods*2*math.pi, n_samples)
        y = amplitude * np.sin(frequency*X + phase_shift) * np.exp(-1*decay*X)
        y = y + np.random.normal(loc=0.0, scale=err_std, size=y.shape)
        dataset = SimulatedDataset(X,y)
        return dataset

class SimulatedClassification:

    @staticmethod
    def blobs(n_features=2, centers=2, cluster_std=1, n_samples=1000, random_state=None):
        
        X, y = datasets.make_blobs(n_samples=n_samples, 
                                   n_features=n_features, 
                                   centers=centers,
                                   cluster_std=cluster_std)

        dataset = SimulatedDataset(X,y)
        return dataset





              