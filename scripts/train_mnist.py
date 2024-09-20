import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from models import MNISTDendFCNet
from utils import trainer

parser = argparse.ArgumentParser(description='Arguments for DendriticLinear MNIST Training')
parser.add_argument('--experiment_name', help='Name of Experiment being Launched', required=True, type=str)
parser.add_argument('--work_dir', help='Working Directory for results storage', default="work_dir", type=str)
parser.add_argument('--resolution', help='Temporal resolution of dendrites', required=True, type=int)
parser.add_argument('--log_weights', action=argparse.BooleanOptionalAction)
parser.add_argument('--log_weights_iters', default=0, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--learning_rate', default=0.0008, type=float)
parser.add_argument('--num_training_steps', default=2500, type=int)
parser.add_argument('--warmup_steps_proportion', default=0.1, type=float)
parser.add_argument('--evaluation_steps', default=50, type=int)
parser.add_argument('--early_stopping_persistance', default=5, type=int)
parser.add_argument('--optimizer', default="adam", choices=("adam", "sgd"), type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--device', default="cuda", type=str)
args = parser.parse_args()

print("Training Arguments")
print(args)

### Define Training Parts ###
model = MNISTDendFCNet(resolution=args.resolution).to(args.device)
train = MNIST('datasets', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                       ]))

test = MNIST('datasets', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

optimizer_dict = {"adam": torch.optim.Adam, 
                  "sgd": torch.optim.SGD}

optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

### Train Model ###
model, training_log, weight_log = trainer(model, 
                                          train_set=train,
                                          test_set=test, 
                                          training_iterations=args.num_training_steps,
                                          warmup_iteration_proportion=args.warmup_steps_proportion,
                                          evaluation_iterations=args.evaluation_steps,
                                          batch_size=args.batch_size, 
                                          optimizer=optimizer, 
                                          loss_fn=loss_fn,
                                          workers=args.num_workers,
                                          device=args.device,
                                          log_weights=args.log_weights,
                                          log_weights_iterations=args.log_weights_iters,
                                          early_stopping_persistance=args.early_stopping_persistance)

### Store Results Paths ###
path_to_experiment = os.path.join(args.work_dir, args.experiment_name)
path_to_log = os.path.join(path_to_experiment, "log.pkl")
path_to_weight_log = os.path.join(path_to_experiment, "weight_log.pkl")
path_to_training_details = os.path.join(path_to_experiment, "training_args.pkl")
path_to_model_store = os.path.join(path_to_experiment, "model.bin")

if not os.path.exists(path_to_experiment):
    os.mkdir(path_to_experiment)
    
### Save Training Logs ###
with open(path_to_log, "wb") as f:
    pickle.dump(training_log, f, protocol=pickle.HIGHEST_PROTOCOL)

### Save Training Arguments Information ###
with open(path_to_training_details, "wb") as f:
    pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)

### Save Weight Logs ###
if args.log_weights:
    with open(path_to_weight_log, "wb") as f:
        pickle.dump(weight_log, f, protocol=pickle.HIGHEST_PROTOCOL)

### Save Model ###
torch.save(model.state_dict(), path_to_model_store)









