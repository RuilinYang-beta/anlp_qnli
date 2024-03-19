import torch
from sklearn.metrics import accuracy_score, f1_score
from statics import DEVICE  

def evaluate_model(model, dataset, criterion, 
                    isRNN=False, isFFNN=False, isTransformer=False, 
                    **kwargs  # model-specific hyperparams for forward method
                    ):
    """
    Args:
        model: model to evaluate
        h_0: initial hidden state
        dataset: an instance of `Dataset` class
        criterion: loss function    
    """

    model_flag = sum([isRNN, isFFNN, isTransformer])
    assert model_flag == 1, "Please specify one and only one model type."

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    correct = 0 #
    total = 0 #

    num_samples = 0

    # for x, y in dataset:
    for idx, (x, y) in enumerate(dataset):    
        x = x.to(DEVICE) #
        y = y.to(DEVICE) #

        with torch.no_grad():
            # call the model-specific forward method
            if isRNN: 
                output, _ = model(x)

            if isFFNN:
                output = model(x)

            if isTransformer:
                output = model(x)
                
            loss = criterion(output, y)
            total_loss += loss.item()
            predicted = torch.argmax(output, dim=0)

            # Calculate predictions
            correct += (predicted == y).sum().item() #
            if y.dim() == 0:
                total += 1  # Increment total by 1 if y is a scalar tensor
            else:
                total += y.size(0)  # Increment total by the size of the first dimension if y is a regular tensor

            all_preds.append(predicted.cpu())
            all_labels.append(y.cpu())

        # Number of pairs for computation of average loss function
        num_samples += 1

    avg_loss = total_loss / num_samples

    accuracy = correct / total
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, f1_macro
