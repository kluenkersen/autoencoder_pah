import pandas as pd
from torch.utils.data import DataLoader

class CSVLoader(object):
    '''
        loads csv file and gives you timeframe back
    '''
    def __init__(self, data_path, timeframe=5):
        self.data = pd.read_csv(data_path, header=None)
        self.timeframe = timeframe
    
    def __len__(self):
        '''
            Number of values in the dataframe
        '''
        return (self.data.shape[0] - self.timeframe)
        
    def __getitem__(self, index):
        '''
            Get X rows out of the csv file reshaped to a numpy array in one row
        '''
        result = self.data[index: index + self.timeframe]
        result = result.values.reshape(1, self.timeframe * self.data.shape[1])
        return result

def get_loader(data_path='data/autoencoder_v1_PAH3DEEUR_1 Min_Bid_2008.10.21_2018.10.27', 
               batch_size = 64, timeframe=5):
    '''
        loads the dataloader object from the dataset
    '''
    loader = CSVLoader(data_path=data_path, timeframe = timeframe)
    data_loader = DataLoader(dataset=loader, batch_size = batch_size)
    return data_loader
