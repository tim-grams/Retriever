import torch.nn as nn


class RankNet(nn.Module):
    ''' A class to create pair wise RankNet models.

    Attributes:
        num_features (): 

    Methods:
    forward(input1, input2)
        INSERT_DESCRIPTION
    predict_proba(input_):
        INSERT_DESCRIPTION
        
    '''

    def __init__(self, num_features):
        ''' Constructs RankNet object. 
        
        Args: 
            num_features ():

        '''
        super(RankNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1))

        self.output = nn.Sigmoid()

    def forward(self, input1, input2):
        ''' . 
        
        Args: 
            input1 ():
            input2 ():

        Returns:
            prob ():

        '''
        s1 = self.model(input1)
        s2 = self.model(input2)
        diff = s1 - s2
        prob = self.output(diff)

        return prob

    def predict_proba(self, input_):
        ''' . 
        
        Args: 
            input_ ():

        Returns:
            ():

        '''
        return self.model(input_)
