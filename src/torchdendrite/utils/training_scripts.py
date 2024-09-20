import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from .logger import WeightLogger

def trainer(model, 
            train_set,
            test_set, 
            training_iterations,
            warmup_iteration_proportion,
            evaluation_iterations,
            batch_size, 
            optimizer, 
            loss_fn,
            leaky=False,
            workers=1,
            device=None,
            log_weights=True,
            log_weights_iterations=25,
            early_stopping_persistance=5,
            early_stopping_threshold=0.0005):
    
    ### If we want to log weights create storage dictionary ###
    if log_weights:
        weight_logger = WeightLogger(model)

    ### Set Device ###
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    ### Prepare DataLoaders ###
    print("Number of Training Samples:", len(train_set))
    print("Number of Testing Samples:", len(test_set))

    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=workers, 
                              pin_memory=True)

    test_loader = DataLoader(test_set, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=workers, 
                             pin_memory=True)

    ### Create logs to Store Training Performance ###
    training_logs = {"completed_steps": [], 
                     "training_loss": [], 
                     "testing_loss": []}
    train_loss, test_loss = [], []
    compute_accuracy = False 
    
    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
        compute_accuracy = True
        training_logs["training_acc"] = []
        training_logs["testing_acc"] = []
        train_acc, test_acc = [], []


    ### Training Loop ###
    train = True
    completed_steps = 0
    best_eval_loss = np.inf
    early_stopping_counter = 0
    progress_bar = tqdm(range(training_iterations))

    ### Scheduler ###
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                   num_training_steps=training_iterations, 
                                                   num_warmup_steps=int(training_iterations*warmup_iteration_proportion))
    
    while train:

        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            ### For Non Leaky Shape is [B, 10], so compute loss normally ###
            if not leaky:
                pred = model(X)
                loss = loss_fn(pred, y)
                train_loss.append(loss.item())
                
                ### Compute Accuracy ###
                if compute_accuracy:
                    predictions = torch.argmax(pred, axis=1)
                    accuracy = (predictions == y).sum() / (len(predictions))
                    train_acc.append(accuracy.item())

            ### for Leaky shape is [T, B, 10], we need to add up loss at every step against the membrane potential ###
            else:
                spk_rec, mem_rec = model(X)
                loss = 0
                for step in range(model.num_steps):
                    loss  = loss + loss_fn(mem_rec[step], y)
                train_loss.append(loss.item())

                ### To compute accuracy, just sum up the spikes and take the index with the msot spikes ###
                if compute_accuracy:
                    accumulated_spikes = spk_rec.sum(dim=0)
                    predictions = torch.argmax(accumulated_spikes, axis=1)
                    accuracy = (predictions == y).sum() / (len(predictions))
                    train_acc.append(accuracy.item())
            
            ### Update Model ###
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            ### Evaluate every evaluation_iterations steps ###
            if (completed_steps+1) % evaluation_iterations == 0:
            
                model.eval()
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
        
                    ### Inference and Compute Loss ###

                    if not leaky:
                        with torch.no_grad():
                            pred = model(X)
                        loss = loss_fn(pred, y)
                        test_loss.append(loss.item())
                        
                        ### Compute Accuracy ###
                        if compute_accuracy:
                            predictions = torch.argmax(pred, axis=1)
                            accuracy = (predictions == y).sum() / (len(predictions))
                            test_acc.append(accuracy.item())

                    else:
                        spk_rec, mem_rec = model(X)
                        loss = 0
                        for step in range(model.num_steps):
                            loss  = loss + loss_fn(mem_rec[step], y)
                        test_loss.append(loss.item())
                        
                        if compute_accuracy:
                            accumulated_spikes = spk_rec.sum(dim=0)
                            predictions = torch.argmax(accumulated_spikes, axis=1)
                            accuracy = (predictions == y).sum() / (len(predictions))
                            test_acc.append(accuracy.item())
                    
                ### Average Loss for Iteration ###
                avg_train_loss = np.mean(train_loss)
                avg_test_loss = np.mean(test_loss)

                training_logs["completed_steps"].append(completed_steps)
                training_logs["training_loss"].append(avg_train_loss)
                training_logs["testing_loss"].append(avg_test_loss)

                print(f"Completed Steps {completed_steps}/{training_iterations}")
                print("Training Loss:", avg_train_loss)
                print("Testing Loss:", avg_test_loss)
                
                train_loss, test_loss = [], []
                
                if compute_accuracy:
                    avg_train_acc = np.mean(train_acc)
                    avg_test_acc = np.mean(test_acc)
                    training_logs["training_acc"].append(avg_train_acc)
                    training_logs["testing_acc"].append(avg_test_acc)
                    print("Training Acc:", avg_train_acc)
                    print("Testing Acc:", avg_test_acc)
                    train_acc, test_acc = [], []

                ### Early Stopping Check ###\

                if abs(avg_test_loss - best_eval_loss) < early_stopping_threshold:
                    early_stopping_counter += 1
                    if early_stopping_counter == early_stopping_persistance:
                        train = False
                        print("Completed Training: Early Stopping")
                        break
                else:
                    early_stopping_counter = 0
                    best_eval_loss = avg_test_loss
    

            ### Iterate and Update Progress Bar ###
            completed_steps += 1
            progress_bar.update(1)
        
            ### Log Weights ###
            if log_weights:
                if completed_steps % log_weights_iterations == 0:
                    weight_logger.log_weights(model)

            ### Completed Training ###
            if completed_steps >= training_iterations:
                train = False
                print("Completed Training!")
                break
            
    if log_weights:
        return model, training_logs, weight_logger.weight_log
    else:
        return model, training_logs, None
            
            
                
            
            

    
        

    
    






















