import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet101

# Defintion for different models

# Used below
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    

# basic CNN - CNN on middle frame but with resnet
class BasicCNN(nn.Module):
    def __init__(self, num_frames=10, num_classes=2):
        print("Running BasicCNN")
        super().__init__()
        self.num_classes = num_classes
        self.resnet = resnet101(pretrained=True)

        #  eplace final linear layer with identity
        self.resnet.fc = Identity() #

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        if x.dim() == 5:
            NUM_BATCHES, NUM_FRAMES, C, H, W = x.shape

            x_frame = x[: , NUM_FRAMES // 2, : , : , :]

            x_frame = self.resnet(x_frame)

            # flatten all dimensions except batch
            x_frame = torch.flatten(x_frame, 1) 

            x_frame = F.relu(self.fc1(x_frame))
            x_frame = self.fc2(x_frame)

        return x_frame


# basic CNN - CNN on frame near end but with resnet
class FrameVoting(nn.Module):
    def __init__(self, num_frames=10, num_classes=2):
        print("Running FrameVoting")
        super().__init__()
        self.num_classes = num_classes

        self.resnet = resnet101(pretrained=True)

        #  replace final linear layer with identity
        self.resnet.fc = Identity() #

        # make arrays  fully connected layers - one for each frame
        self.fc1s = []
        self.fc2s = []

        for _ in range(num_frames):

            fc1 = nn.Linear(2048, 128)
            fc2 = nn.Linear(128, num_classes)

            self.fc1s.append(fc1)
            self.fc2s.append(fc2)

        self.fc1s = nn.ModuleList(self.fc1s)
        self.fc2s = nn.ModuleList(self.fc2s)

        self.fc_final = nn.Linear(num_frames * num_classes, num_classes)
            

    def forward(self, x):
        if x.dim() == 5:
            NUM_BATCHES, NUM_FRAMES, C, H, W = x.shape

        frame_outputs = torch.empty(NUM_BATCHES, NUM_FRAMES, self.num_classes)

        device = 'cpu'
        if torch.cuda.is_available(): 
            device = torch.device("cuda:0") 
            frame_outputs = frame_outputs.to(device)


        for t in range(NUM_FRAMES):
            x_frame = x[: , t, : , : , :]

            x_frame = self.resnet(x_frame)

            # flatten all dimensions except batch
            x_frame = torch.flatten(x_frame, 1).to(device)

            x_frame = F.relu(self.fc1s[t](x_frame))
            x_frame = self.fc2s[t](x_frame)

            frame_outputs[:, t, :] = x_frame


        # now take all frame outputs and do a final linear layer
        frame_outputs = frame_outputs.reshape(NUM_BATCHES, -1)

        return self.fc_final(frame_outputs)



# Based on https://github.com/pranoyr/cnn-lstm/blob/master/models/cnnlstm.py
# LSTM + CNN
# CURRENTLY Just feeding last hidden layer output to FC layer, instead of all frame lstm outputs 
class CNN_LSTM(nn.Module):
    def __init__(self, num_frames=10, hidden_size=200, num_classes=2):
        print("Running CNN_LSTM")
        super(CNN_LSTM, self).__init__()
        self.resnet = resnet101(pretrained=True) # pretrained resnet CNNN

        self.resnet.fc = Identity() # replace final linear layer with identity
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=3)
        self.fc1 = nn.Linear(hidden_size * 1, 128)
        self.fc2 = nn.Linear(128, num_classes)
       
    def forward(self, i): 
        x = i.view(-1, i.shape[2], i.shape[3], i.shape[4]) # original code with reshaping
     
        x = self.resnet(x)

        # reseparating x's batch and frames (having just been through CNN) back. 
        # Now, x.shape = (num_batches, num_frames, channels * width * height)
        x = x.view(i.shape[0], i.shape[1], -1)
        
        full_output, last_hidden = self.lstm(x)

        x = full_output[:, -1, :]

        # x shape is (num_batches, lsmt output) 
        x = x.reshape(i.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    


