import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np


# Defines loss function and trains model (returns trained model -- maybe train in place)
def train(model, train_loader, num_epochs):

    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
            #print(f"accuracy = {correct / predicted.shape[0]}")

            #print("outputs: ", outputs )
            #print("predicted: ", predicted)
            #print("actual labels: ", labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses = np.append(losses, loss.item())

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

            #print("losses = ", losses)
    print('Finished Training')
    return losses


# Similar to above but uses learning rate decay based on epoch
# Defines loss function and trains model (returns trained model -- maybe train in place)
def train2(model, train_loader, num_epochs, device):
    print("train 2 active")
    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # every 15 epochs, learning rate is multiplied by .1
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)


    losses = np.array([])

    print("Train Model on training data...")
    for epoch in range(num_epochs):
        running_loss = 0
        torch.cuda.empty_cache()
        for i, batch_data in enumerate(train_loader, 0):

            # Format batch
            #real_cpu = batch_data.to(device)
            #b_size = real_cpu.size(0)
            #label = torch.full((b_size,), 0, device=device).type(torch.float32)  

            inputs, labels = batch_data

            inputs = inputs.to(device)
            labels = labels.to(device)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            _ , predicted = torch.max(outputs.data, 1)

            correct = (predicted == labels).sum().item()

            print("outputs: ", outputs )
            print("predicted: ", predicted)
            print("actual labels: ", labels)
           
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses = np.append(losses, loss.item())

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

            #print("losses = ", losses)
            torch.cuda.empty_cache()

        lr_scheduler.step()
        
    print('Finished Training')
    return losses