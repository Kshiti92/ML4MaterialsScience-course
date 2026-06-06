import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # CONVOLUTIONAL LAYERS
        # 3 input image channel (RGB)+ 1 for oppacity; 6 output size and 10 (10px*10px) Kernel size seem appropraite for the microstructure size
        self.conv1 = nn.Conv2d(4, 6, 11)
        # Max pooling over a (2, 2) window; Kernel size 2, Stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # input for conv2 must be equal to the output channel size of last conv layer (conv1)
        # output size 16 and kernel size 8
        self.conv2 = nn.Conv2d(6, 16, 8)

        ## FULLY CONNCECTED LAYERS
        # an affine operation: y = Wx + b
        # input layer 16*22*22 and output layer 120
        self.fc1 = nn.Linear(16 * 19 * 19, 20)  # 18*18 from image dimension after conv1,pool,con2,pool see below
        # again input layer has to match last output layer
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20, 1)
        # again input layer has to match last output layer
        #output classes will be 1 because we want to have a single float at the end

        """
        https://stackoverflow.com/questions/66337378/mat1-and-mat2-shapes-cannot-be-multiplied
        https://www.youtube.com/watch?v=pDdP0TFzsoQ


        Initial image size is (Batchsize,Input Channels,x1-Dimension,x2-Dimension)
        Initial image size is (4,3,100,100)
        Batch size is 4, our image channels are 3 (RGB) and our picture Dimensions are 100x100

        After first convolution image size is (4,6,91,91)
        the output layer is 6 (chosen in first conv. layer) and the dimensions are reduced by 9 to 90x90
            due to the kernel size of 11x11
            Image size: (((W - K + 2P)/S) + 1) = (((100-11+2*0)/1)+1) = 90
              Here W = Input size K = Filter size S = Stride P = Padding
        The pooling reduces the image to (4,6,48,48)
            (((W - K + 2P)/S) + 1) = (((90-2+2*0)/2)+1) = 45
        After second convolution image size is (4,16,44,44)
            (((W - K + 2P)/S) + 1) = (((45-8+2*0)/1)+1) = 38
        Another pooling layer reduces the image to (4,16,22,22)
            (((W - K + 2P)/S) + 1) = (((38-2+2*0)/2)+1) = 19
        """


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = torch.flatten(x) # flatten to pass to fully connected layer
        x = x.view(-1,16*19*19)
       # x = x.reshape(x.size(),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation function at the end
        return x


net = Net()
#net.to(device)  # if theres an error with the "device" look above in the first code snippet