import torch as pt
from utils import logistic_fun, binomial_coeff, calculate_weights
from lossClasses import MyCrossEntropy, MyRootMeanSquare
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def generate_data(N:int, M:int, D=5, add_noise=True):
    p  = sum(binomial_coeff(D + m - 1, m) for m in range(M + 1))
    print(p)
    w = calculate_weights(p)
    print(w.size())
    X = pt.empty(N, D).uniform_(-5.0, 5.0)
    y = logistic_fun(w, M, X)
    if add_noise:
        y+= pt.randn(y.size())
    targets = (y >= 0.5).int()
    return X, targets

def fit_logistic_sgd(X:pt.Tensor, targets:pt.Tensor, loss_type:str, M:int, lr=0.001, batch_size=32, epochs=1000):
     
    """
    Use loss_type = 'CE' for MyCrossEntropy or loss_type = 'RMS' for MyRootMeanSquare
    """

    dataset = TensorDataset(X, targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    N, D = X.size()
    p  = sum(binomial_coeff(D + m - 1, m) for m in range(M + 1))
    weights = pt.randn(p, requires_grad=True)
    optimizer = optim.SGD([weights], lr=lr)
    match loss_type:
        case 'CE':
            loss_function = MyCrossEntropy()
        case 'RMS':
            loss_function = MyRootMeanSquare()
        case _:
            assert False #Loss not supported

    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = logistic_fun(weights, M, batch_X)
            loss = loss_function(predictions.float(), batch_y.float())
            loss.backward()
            optimizer.step()

        if epoch%50 ==0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():4f}")

    train_pred = logistic_fun(weights, M, X)
    train_pred_labels = (train_pred >= 0.5).int()
    accuracy = (train_pred_labels == targets).float().mean()
    print(f"Train Accuracy: {accuracy.item() :.4f}")
    
    return weights.detach()



        



if __name__ == '__main__':

    pt.manual_seed(42)
    
    train_X, train_targets = generate_data(N=200, M=2 )
    print(train_X, train_targets)
    test_X, test_targets = generate_data(N=100, M=2, add_noise=False)
    print(sum(test_targets))
    Ms = [1,2,3]
    Ls = ["CE", "RMS"]
    for loss_type in Ls:
        print(f"==========Loss {loss_type}==========")
        for m in Ms:
            print(f"---------- M = {m}-------------")
            w_hat = fit_logistic_sgd(train_X, train_targets, loss_type, m)
            #print(w_hat)
            test_pred =logistic_fun(w_hat, m, test_X)
            
            match loss_type:
                case 'CE':
                    test_loss_fn = MyCrossEntropy()
                case 'RMS':
                    test_loss_fn = MyRootMeanSquare()

            test_loss = test_loss_fn(test_pred.float(), test_targets.float())

            test_pred_labels = (test_pred >= 0.5).int()

            accuracy = (test_pred_labels == test_targets).float().mean()

            print(f"Test Loss: {test_loss.item()}")
            print(f"Test Accuracy: {accuracy.item() :.4f}")


