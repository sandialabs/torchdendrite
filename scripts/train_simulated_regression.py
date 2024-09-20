import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle
import matplotlib.pyplot as plt
from models import DendFCNetRegressor
from utils import trainer, SimulatedRegression, plot_results

parser = argparse.ArgumentParser(description='Arguments for Simulated Data Training')
parser.add_argument('--experiment_name', help='Name of Experiment being Launched', required=True, type=str)
parser.add_argument('--dataset', help="Which simulated data do you want?", required=True, choices=('simple_sinusoid', 'linear'), type=str)
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
model = DendFCNetRegressor(resolution=args.resolution).to(args.device)

datasets = {"simple_sinusoid":  SimulatedRegression.simple_sinusoid, 
            "linear": SimulatedRegression.linear}

if args.dataset == "simple_sinusoid":
    train = SimulatedRegression.simple_sinusoid(num_periods=2, n_samples=2500)
    test = SimulatedRegression.simple_sinusoid(num_periods=2, n_samples=250)
elif args.dataset == "linear":
    train = SimulatedRegression.linear(n_samples=2500)
    test = SimulatedRegression.linear(n_samples=250)
else:
    raise ValueError(f"{args.dataset} Unknown")

optimizer_dict = {"adam": torch.optim.Adam, 
                  "sgd": torch.optim.SGD}

optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.learning_rate)

loss_fn = torch.nn.MSELoss()

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
path_to_plot = os.path.join(path_to_experiment, "results.png")
path_to_log = os.path.join(path_to_experiment, "log.pkl")
path_to_weight_log = os.path.join(path_to_experiment, "weight_log.pkl")
path_to_model_store = os.path.join(path_to_experiment, "model.bin")
path_to_training_details = os.path.join(path_to_experiment, "training_args.pkl")

if not os.path.exists(path_to_experiment):
    os.mkdir(path_to_experiment)

### Inference Models
with torch.no_grad():
    y_pred = model(torch.tensor(test.x).type(torch.FloatTensor).reshape(-1,1).to(args.device)).cpu().numpy()
    
### Plot Results ###
plot_results(test.x, test.y, y_pred, path_to_plot, type="regression")

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

