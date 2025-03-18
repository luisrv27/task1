import torch as pt
from utils import logistic_fun, binomial_coeff, generate_data
from metrics import F1Score, Accuracy
from lossClasses import MyCrossEntropy, MyRootMeanSquare
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn



def fit_logistic_sgd(X:pt.Tensor, targets:pt.Tensor, loss_type:str, M:int, lr=0.1, batch_size=32, epochs=1000, verbose=True):
     
    """
    Use loss_type = 'CE' for MyCrossEntropy or loss_type = 'RMS' for MyRootMeanSquare
    """

    dataset = TensorDataset(X, targets)
    
    N, D = X.size()
    p  = sum(binomial_coeff(D + m - 1, m) for m in range(M + 1))
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    weights = pt.randn(p, device=device, requires_grad=True)
    optimizer = optim.SGD([weights], lr=lr)
    loss_dict = {'CE': MyCrossEntropy(), 'RMS': MyRootMeanSquare()}
    loss_function = loss_dict.get(loss_type, None)
    if not loss_function:
        raise ValueError("Loss not supported")
    train_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
    for epoch in range(epochs):
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = logistic_fun(weights, M, batch_X)
            loss = loss_function(predictions.double(), batch_y.double())
            loss.backward()
            optimizer.step()

        if verbose and epoch%50 ==0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():4f}")

    train_pred = logistic_fun(weights, M, X)
    train_pred_labels = (train_pred >= 0.5).int()
    accuracy = Accuracy(targets,train_pred_labels)
    f1 = F1Score(targets,train_pred_labels)
    if verbose:
        print(f"Train Accuracy: {accuracy :.4f}")
        print(f"Train F1 Score: {f1 :.4f}")
    
    return weights.detach()



        



if __name__ == '__main__':

    #pt.manual_seed(42)
    
    train_X, train_targets = generate_data(N=200, M=2, add_noise=True)

    test_X, test_targets = generate_data(N=100, M=2, add_noise=False)
    
    Ms = [1,2,3]
    Ls = ["CE", "RMS"]
    metrics = {"CE":{"accuracy":[],"f1":[]}, "RMS":{"accuracy":[],"f1":[]}}
    for loss_type in Ls:
        print(f"==========Loss {loss_type}==========")
        for m in Ms:
            print(f"---------- M = {m}-------------")
            w_hat = fit_logistic_sgd(train_X, train_targets, loss_type, m)

            test_pred =logistic_fun(w_hat, m, test_X)
            loss_dict = {'CE': nn.BCELoss(), 'RMS': MyRootMeanSquare()}
            loss_function = loss_dict.get(loss_type, None)
            if not loss_function:
                raise ValueError("Loss not supported")

            test_loss = loss_function(test_pred.double(), test_targets.double())

            test_pred_labels = (test_pred >= 0.5)

            metrics[loss_type]["accuracy"].append(Accuracy(test_targets.double(),test_pred_labels.double()))
            metrics[loss_type]["f1"].append(F1Score(test_pred_labels, test_targets))

    print("==============Metrics=================")
    for loss in metrics.keys():
        print(f"---------- Loss {loss}----------")
        for metric in metrics[loss].keys():
            for m in Ms:
                print(f"{metric} Score Test for m={m}: {metrics[loss][metric].pop():4f}")
