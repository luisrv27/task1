import torch as pt
from task import fit_logistic_sgd, generate_data,Accuracy,F1Score, logistic_fun, MyCrossEntropy
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import torch.nn.functional as F

if __name__ == "__main__":
    
    pt.manual_seed(42)
    train_X, train_targets = generate_data(N=200, M=2, add_noise=True)
    val_X, val_targets = generate_data(N=100, M=2, add_noise=True) 
    test_X, test_targets = generate_data(N=100, M=2, add_noise=False)

    data_loader = DataLoader(TensorDataset(val_X,val_targets), batch_size=32)
    num_experts = 5

    Ms = pt.arange(1,num_experts+1)
    gating_weights = pt.full(Ms.size(),1/Ms.size()[0],requires_grad=True)
    lr = 1e-3
    num_epochs = 500
    optimizer = optim.SGD([gating_weights], lr=lr)
    
        # Initialize experts and gating weights (single set of weights)
    print("Training experts")
    experts = []
    for m in Ms:
        sys.stdout.write(f"\rexpert {m}/{num_experts}")
        sys.stdout.flush()
        experts.append((fit_logistic_sgd(X=train_X,targets=train_targets,loss_type="CE",M=m,verbose=False),m))

    # Training loop
    loss_fn = MyCrossEntropy()
    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs, targets = batch
            expert_outputs=[]
            for expert in experts:
                pred = logistic_fun(w=expert[0],M=expert[1],x=inputs)
                lbl = (pred >=0.5)
                expert_outputs.append(loss_fn(pred.double(),targets.double()).item() - (Accuracy(targets,lbl) +F1Score(targets,lbl)))
            
            loss = sum(gating_weights[i] * expert_outputs[i] for i in range(num_experts))   
            
            loss.backward()  
            
            optimizer.step()
            gating_weights.data.div_(gating_weights.data.sum())

            optimizer.zero_grad()

        sys.stdout.write(f"\rRunning epoch: {epoch + 1} / {num_epochs}, Loss: {loss.item():4f}") 
        sys.stdout.flush()

    best_M = Ms[gating_weights.argmax()].item()
    print(f"\nbest M: {best_M}")
    

    test_pred =logistic_fun(experts[gating_weights.argmax()][0], best_M, test_X)
    test_pred_labels = (test_pred >= 0.5)
    print(f"Accuracy Score: {Accuracy(test_targets.double(),test_pred_labels.double()):4f}")
    print(f"F1 Score: {F1Score(test_targets.double(),test_pred_labels.double()):4f}")
 