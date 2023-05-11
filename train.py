import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np


# Similar to above but uses learning rate decay based on epoch
# Defines loss function and trains model
def train(model, train_loader, num_epochs, device):
    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # every 30 epochs, learning rate is multiplied by .2
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)


    losses = np.array([])

    print("Train Model on training data...")
    for epoch in range(num_epochs):
        running_loss = 0
        torch.cuda.empty_cache()
        for i, batch_data in enumerate(train_loader, 0):
  
            inputs, labels = batch_data

            # Assign tensors to cpu or gpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
           
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