import torch
from tqdm import tqdm

# Testing (with validation data)
def test(model, test_loader, device):
    correct = 0
    total = 0

    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data

            images = images.to(device)

            # calculate outputs by running images through the network
            outputs = model(images).to(device)
            labels = labels.to(device)


            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.to(device)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            torch.cuda.empty_cache()
    print(
        f'Accuracy of the network on test images: {100 * correct // total} %')

