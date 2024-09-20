class WeightLogger:
    def __init__(self, model):
        self._create_logging_dict(model)
        
    def _create_logging_dict(self, model):
        print("Initialized Weight Logger")
        
        self.weight_log = {}
        for name, param in model.named_parameters():
            self.weight_log[name] = [param.cpu().detach().numpy()]

    def log_weights(self, model):
        for name, param in model.named_parameters():
            self.weight_log[name].append(param.cpu().detach().numpy())