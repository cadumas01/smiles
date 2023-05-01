import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np



# Defines loss function and trains model (returns trained model -- maybe train in place)
def train(model, train_loader, num_epochs):

    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)

    losses = np.array([])

    print("Train Model on training data...")
    for epoch in range(num_epochs):
         running_loss = 0
         for i, batch_data in enumerate(train_loader, 0):

            inputs, labels = batch_data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            _ , predicted = torch.max(outputs.data, 1)

            correct = (predicted == labels).sum().item()
            print(f"accuracy = {correct / predicted.shape[0]}")

            print("outputs: ", outputs )
            print("predicted: ", predicted)
            print("actual labels: ", labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np.append(losses, loss)

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return losses