import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import FeedForwardNN

def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        #todo
        for inputs,labels in dataloader:
            #moves inputs and labels to cpu or gpu
            inputs, labels = inputs.to(device), labels.to(device)
            #zero gradients before forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            #loss
            loss = criterion(outputs,labels)
            #backprop
            loss.backward()
            optimizer.step()

            #track metrics (running loss)
            running_loss += loss.item() * inputs.size(0)
            hehe, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            #compute epoch metrics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_accuracy = correct_predictions / total_samples

            #print progress
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            print("Training finished")

def main():
    dataset = utils.load_data('train.json')
    vocab = utils.build_vocabulary(dataset, notation='ORIGINAL')
    inputs,labels = utils.xy_pair_generator(dataset, 'ORIGINAL', vocab)

    # Create a DataLoader??
    batch_size = 32 #??
    shuffle = True #???
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) #???

    vocab_size = len(vocab)
    embedding_dim = 0
    hidden_size = 150
    output_size = 3
    model = FeedForwardNN(vocab_size, embedding_dim, hidden_size, output_size)

if __name__ == "__main__":
    main()

