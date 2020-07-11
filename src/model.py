from dataPreparation import *

NumberOfUniqueClasses = 2 # !IMPORTANT! change this base number of classes

# MLP to divide output from alexnet weights
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256*6*6, 100),
            nn.ReLU(),
            nn.Linear(100,NumberOfUniqueClasses) 
        )
        

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6)
        x = self.layers(x)
        return x