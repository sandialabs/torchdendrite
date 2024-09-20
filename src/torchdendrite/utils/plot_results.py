import matplotlib.pyplot as plt

def plot_results(X, y, y_pred, path_to_plot, type="regression"):

    if type == "regression":

        if len(X.shape) > 1:
            assert X.shape[-1] == 1, "Can only plot regression for single input feature"

        plt.scatter(X, y, s=2, label="GT")
        plt.scatter(X, y_pred, s=2, label="Pred")
        plt.title("Test Data Inference")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig(path_to_plot, dpi=250)

    else:
        
        assert X.shape[-1] <= 2, "Can only plot classifications for upto 2 input features"

        for c in np.unique(y_pred):
            selected = (y_pred == c)
            plt.scatter(X[selected,0], X[selected,1], label=c)
            
        plt.savefig(path_to_plot, dpi=250)
            
        
        