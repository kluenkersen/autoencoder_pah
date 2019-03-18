import pandas as pd
import numpy as np

def create_X_y(filename, path='data', timeframe=30, boarder=0.2, loss_ratio=3):
    """[This can take a while :-), we are looping over the hole csv file to create new files.
        Created will be X_filename and y_filename.
        X_filename: ]
        
    Arguments:
        filename {[string]} -- [the file name with path to csv file.]
        path {[string]} -- [default = data, set the dirctory where the file is stored]
        timeframe {[int]} -- [default is set to 30, means that we check 30 minutes in the future if we reach the
                               set boarder. E.g. Dataframe[i:i+30] are checked if the boarder is hit]
        boarder {[float]} -- [boarder describes how many pips we want to go in minus befor hiting our target.
                                It is to define if the trade would be successfull]
        loss_ratio {[int]} -- [Loss_ratio describes how many times * we want to get out of the trade. If
                                the boarder is set to 0.2 and the loss ratio = 2 than we are looking for a 0.4
                                take profit and a 0.2 stop loss. Is loss ratio = 3 than take porfit = 0.6 and
                                stop loss = 0.2. If loss_ratio is set to 10 we will run over 10-2 stimes. 
                                Therefore it creates then 8 entries for y]

    Returns:
        [dict] -- [Filename of generated files]
    """
    X =  create_X(filename, path, timeframe)
    y = create_y(filename, path, timeframe, boarder, loss_ratio)
    return {'X': X, 'y': y}
    

def create_X(filename, path='data', timeframe=30):
    """[Changes the csv file to X_csv file to display only the changes values like high-low or open-close
        instead of the actuals. Result file will be stored in the path folder]
        
    Arguments:
        filename {[string]} -- [the file name with path to csv file.]
        path {[string]} -- [default = data, set the dirctory/folder where the file is stored]
        timeframe {[int]} -- [default is set to 30, means that we check 30 minutes in the future if we reach the
                               set boarder. E.g. Dataframe[i:i+30] are checked if the boarder is hit]
    
    Returns:
        [string] -- [Filename of generated file]
    """
    # Create X
    df = pd.read_csv(path +'/'+ filename)
    X = pd.DataFrame()
    X['high_low'] = df.High - df.Low
    X['high_open'] = df.High - df.Open
    X['high_close'] = df.High - df.Close
    X['low_open'] = df.Low - df.Open
    X['low_close'] = df.Low - df.Close
    X['close_open'] = df.Close - df.Open
    X['volume'] = df['Volume ']
    X[:-timeframe].to_csv(path + '/X_' + filename, index=None)
    
    return 'X_' + filename

def create_y(filename, path='data', timeframe=30, boarder=0.2, loss_ratio=2):
    """[This can take a while :-), we are looping over the hole csv file to create new files.
        Created will be X_filename and y_filename.
        X_filename: ]
        
    Arguments:
        filename {[string]} -- [the file name with path to csv file.]
        path {[string]} -- [default = data, set the dirctory where the file is stored]
        timeframe {[int]} -- [default is set to 30, means that we check 30 minutes in the future if we reach the
                               set boarder. E.g. Dataframe[i:i+30] are checked if the boarder is hit]
        boarder {[float]} -- [boarder describes how many pips we want to go in minus befor hiting our target.
                                It is to define if the trade would be successfull]
        loss_ratio {[int]} -- [Loss_ratio describes how many times * we want to get out of the trade. If
                                the boarder is set to 0.2 and the loss ratio = 2 than we are looking for a 0.4
                                take profit and a 0.2 stop loss. Is loss ratio = 3 than take porfit = 0.6 and
                                stop loss = 0.2. If loss_ratio is set to 10 we will run over 10-2 stimes. 
                                Therefore it creates then 8 entries for y]

    Returns:
        [string] -- [Filename of generated file]
    """
    # read file
    df = pd.read_csv(path +'/'+ filename)
    df_y = pd.DataFrame()
    # creates y
    for l in range(2,loss_ratio):
        y_ = []
        for k in range(df.shape[0] - timeframe):
            y = 0
            # open value for the next bar current start timeframe (i.open)
            close = df.iloc[k].Close
            # minimum / maximum value for the next timeframe e.g. k + 1(next bar) + 30 minutes
            low = df.iloc[k+1:k+1+timeframe].Low.min()
            high = df.iloc[k+1:k+1+timeframe].High.max()
            lsr = boarder * l
            if((close - low) > lsr or (high - close) > lsr):
                for i in range(1, timeframe):
                    # get current bar / bar range
                    bar = df.iloc[k+1:k+i]
                    # get los and high from start (closeing course fist bar) to check if we hitting the boarder
                    open_low = close - bar.Low.min()
                    high_open = bar.High.max() - close
                    if (high_open > lsr and open_low < boarder):
                        y = 1
                        i = timeframe+1
                    if (open_low > lsr and high_open < boarder):
                        # open - low is bigger than X pips and the boarder was not hit before
                        y = 2
                        i = timeframe+1
            # append y to our array y_ to store it later in pandas dataframe
            y_.append(y)
        print(y_)
        # store y_ in the pandas dataframe
        df_y[l] = pd.DataFrame(y_)[0]
    # save y file
    df_y.to_csv(path +'/y_'+filename, index=None)
    
    return 'y_' + filename