import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101, alexnet

# LOOK AT THIS
# https://discuss.pytorch.org/t/cnn-lstm-implementation-for-video-classification/52018

# Defintion for different models

# basic CNN - CNN on middle frame 
class BasicCNN(nn.Module):
    def __init__(self, num_frames=10, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(112896, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        if x.dim() == 5:
            NUM_BATCHES, NUM_FRAMES, C, H, W = x.shape

            x_frame = x[: , NUM_FRAMES // 2, : , : , :]

            x_frame = self.pool(F.relu(self.conv1(x_frame)))
            x_frame = self.pool(F.relu(self.conv2(x_frame)))
            x_frame = torch.flatten(x_frame, 1) # flatten all dimensions except batch
            x_frame = F.relu(self.fc1(x_frame))
            x_frame = F.relu(self.fc2(x_frame))
            x_frame = self.fc3(x_frame)

        return x_frame
    
# Used below
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    

# basic CNN - CNN on middle frame but with resnet
class BasicCNN2(nn.Module):
    def __init__(self, num_frames=10, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = resnet18(pretrained=True)

        self.resnet.fc = Identity() # replace final linear layer with identity

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        if x.dim() == 5:
            NUM_BATCHES, NUM_FRAMES, C, H, W = x.shape

            x_frame = x[: , NUM_FRAMES // 2, : , : , :]

            x_frame = self.resnet(x_frame)

            x_frame = torch.flatten(x_frame, 1) # flatten all dimensions except batch

            x_frame = F.relu(self.fc1(x_frame))
            x_frame = self.fc2(x_frame)

        return x_frame

# basic CNN - CNN on middle frame but with resnet
class BasicCNN3(nn.Module):
    def __init__(self, num_frames=10, num_classes=2):
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
class BasicCNN3(nn.Module):
    def __init__(self, num_frames=10, num_classes=2):
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

            x_frame = x[: , 4 * NUM_FRAMES // 5 , : , : , :]

            x_frame = self.resnet(x_frame)

            # flatten all dimensions except batch
            x_frame = torch.flatten(x_frame, 1) 

            x_frame = F.relu(self.fc1(x_frame))
            x_frame = self.fc2(x_frame)

        return x_frame


# Based on https://github.com/pranoyr/cnn-lstm/blob/master/models/cnnlstm.py
# LSTM + CNN
class CNN_LSTM(nn.Module):
    def __init__(self, num_frames=10, hidden_size=200, num_classes=2):
        super(CNN_LSTM, self).__init__()
        self.resnet = resnet101(pretrained=True) # pretrained resnet CNNN

        print(self.resnet)

        self.resnet.fc = Identity() # replace final linear layer with identity
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=3)
        self.fc1 = nn.Linear(hidden_size * num_frames, 128)
        self.fc2 = nn.Linear(128, num_classes)
       
    def forward(self, i): 
        # x has combined frame and batch dimensions so that new shape is:
        # x.shape = ()
        x = i.view(-1, i.shape[2], i.shape[3], i.shape[4]) # original code with reshaping
     
        x = self.resnet(x)

        # reseparating x's batch and frames (having just been through CNN) back. 
        # Now, x.shape = (num_batches, num_frames, channels * width * height)
        x = x.view(i.shape[0], i.shape[1], -1)
        
        x, last_hidden = self.lstm(x)

        # Test printing, delete
        print("x after lstm = ", x)
        print("last hidden = ", last_hidden)

        print("x shape after lstm ", x.shape)
        # x shape is (num_batches, lsmt output) 
        # It's okay to pass batched data to fully connected Linear layers
        x = x.reshape(i.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    



class CNN_LSTM2(nn.Module):
    def __init__(self,hidden_size=3,n_layers=2,N_classes=2):
        super(CNN_LSTM2, self).__init__()

        self.hidden_size=hidden_size
        self.num_layers=n_layers

        dim_feats = 4096
  
        self.cnn=models.alexnet(pretrained=True)
        self.cnn.classifier[-1]=Identity()
        self.rnn = nn.LSTM(
            input_size=dim_feats,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True)
        self.n_cl=N_classes
        if(True):
            self.last_linear = nn.Linear(2*self.hidden_size,self.n_cl)
        else:
            self.last_linear = nn.Linear(self.hidden_size,self.n_cl)
            

    def forward(self, x):

        batch_size, timesteps, C,H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        
        c_out = self.cnn(c_in)

        r_out, (h_n, h_c) = self.rnn(c_out.view(-1,batch_size,4096))  

        r_out2 = self.last_linear(r_out)

        return r_out2

# Modified from https://discuss.pytorch.org/t/video-classification-with-cnn-lstm/113413/3
# COLE: I think this is working better tha the rest
class CNN_LSTM3(nn.Module):
    def __init__(self, num_frames=5, hidden_size=100, num_classes=2):
        super(CNN_LSTM3, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 30, 5)

        self.lstm = nn.LSTM(input_size=211680, hidden_size = hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size*num_frames, num_classes)
        
    def forward(self, i):

        # x has combined frame and batch dimensions so that new shape is:
        # x.shape = ()
        x = i.view(-1, i.shape[2], i.shape[3], i.shape[4]) # original code with reshaping
        print("x after viewing and before covn1 ", x.shape)
        x = F.relu(self.conv1(x))
        print("x shape after 1 conv", x.shape)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.AvgPool2d(4)(x)

        # reseparating x's batch and frames (having just been through CNN) back. 
        # Now, x.shape = (num_batches, num_frames, channels * width * height)
        x = x.view(i.shape[0], i.shape[1], -1)

        print("about to go into lstm, x shape", x.shape)

        
        x, _ = self.lstm(x)
        print("after lstm x shape = ", x.shape)

        x = x.reshape(i.shape[0], -1)
        x = self.fc(x)
        return x    
    




# Modifed from CNN_LSTM1
# Based on https://github.com/pranoyr/cnn-lstm/blob/master/models/cnnlstm.py
# LSTM + CNN
class CNN_LSTM4(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_LSTM4, self).__init__()
 
        self.conv1 = nn.Conv2d(3, 10, 5)

        # idea: input size must be (channels x width x height ) from output -> single dimension
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
       
    def forward(self, x_3d):
        print("x_3d shape ", x_3d.shape)
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = F.relu(self.conv1(x_3d[:, t, :, :, :]))  
            x = x.reshape()
            print(f"x[{t}] = self.resnet has shape =", x.shape )

            #out, hidden = self.lstm(x.reshape(x_3d.shape[0], 1 -1hidden)      
            print("out shape ", out.shape)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x 
    

# Modifed from CNN_LSMT3 but mmore complicated
# Modified from https://discuss.pytorch.org/t/video-classification-with-cnn-lstm/113413/3
# COLE: I think this is working better tha the rest
class CNN_LSTM5(nn.Module):
    def __init__(self, num_frames=10, hidden_size=100, num_classes=2):
        super(CNN_LSTM5, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 30, 5)

        self.lstm = nn.LSTM(input_size=211680, hidden_size = hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size * num_frames, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, i):

        # x has combined frame and batch dimensions so that new shape is:
        # x.shape = ()
        x = i.view(-1, i.shape[2], i.shape[3], i.shape[4]) # original code with reshaping
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.AvgPool2d(4)(x)

        # reseparating x's batch and frames (having just been through CNN) back. 
        # Now, x.shape = (num_batches, num_frames, channels * width * height)
        x = x.view(i.shape[0], i.shape[1], -1)
        
        x, _ = self.lstm(x)

        # x shape is (num_batches, lsmt output) 
        # It's okay to pass batched data to fully connected Linear layers
        x = x.reshape(i.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    
    

    # Look at Slow fast model