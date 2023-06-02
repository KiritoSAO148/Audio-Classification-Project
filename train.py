from torch import nn
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
import pandas as pd
import matplotlib.pyplot as plt

def training(model, train_dl, num_epochs, device, val_dl):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    for epoch in range(num_epochs):
        model.eval()
        val_loss = 0.0
        val_correct_prediction = 0
        val_total_prediction = 0
        with torch.no_grad():
            for val_data in val_dl:
                val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

                _, val_prediction = torch.max(val_outputs, 1)
                val_correct_prediction += (val_prediction == val_labels).sum().item()
                val_total_prediction += val_prediction.shape[0]

        avg_val_loss = val_loss / len(val_dl)
        avg_val_acc = val_correct_prediction / val_total_prediction

        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, prediction = torch.max(outputs, 1)

            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        avg_acc = correct_prediction / total_prediction

        train_losses.append(avg_loss)
        train_accuracies.append(avg_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        print(f'Num Batches: {num_batches}, Running Loss: {running_loss}, Correct Prediction: {correct_prediction}'
              f', Total Prediction: {total_prediction}')
        print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.2f}, Accuracy: {avg_acc:.2f}')
        print('---------------------------------------------------------------------------------------------------')

    # Save model
    torch.save(model.state_dict(), 'cnnnet.pt')

    print('Finished Training')
    # Plot loss and accuracy
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train')
    plt.plot(epochs, val_losses, 'r-', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Train')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    num_epochs = 60
    cnn = CNNNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = cnn.to(device)
    print(device)
    print(cnn)

    data_path = 'UrbanSound8K'
    metadata_file = data_path + '/metadata/UrbanSound8K.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    df.head()

    myds = UrbanSoundDataset(df, data_path)

    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    print('---------------------------------------------------------------------------------------------------')
    training(cnn, train_dl, num_epochs, device, val_dl)