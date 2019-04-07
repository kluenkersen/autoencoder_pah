import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.metrics import confusion_matrix

def xgEvaluation(X, y, y_field='4', i_end=50, debug_i=5, threshold_confusionm=0.5):
    # define split range by %
    split = int(X.shape[0]/100*30)
    split = split
    # split train test
    X_train = pd.DataFrame(X[0])[split:].reset_index(drop=True)
    y_train = y[split:].reset_index(drop=True)[y_field]
    X_test = pd.DataFrame((X[0])[:split]).reset_index(drop=True)
    y_test = y[:split].reset_index(drop=True)[y_field]
    y_train = pd.DataFrame(y_train).replace(to_replace=[2], value=1)
    y_test = pd.DataFrame(y_test).replace(to_replace=[2], value=1)

    y_train_ = pd.DataFrame()
    y_train_ = pd.DataFrame()
    X_train_df = pd.DataFrame()
    X_test_df = pd.DataFrame()
    y_columns = y.columns
    for i in range(i_end):
        pct_change = i * 5    
        if(pct_change == 0):
            pct_change = 1
        X_train_ = X_train.pct_change(periods=pct_change)
        X_test_ = X_test.pct_change(periods=pct_change)
        # get again right format for xgboost
        X_test_.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_train_.fillna(0, inplace=True)
        X_test_.fillna(0, inplace=True)
        X_train_ = pd.DataFrame(X_train_, dtype='float32')
        X_test_ = pd.DataFrame(X_test_, dtype='float32')
        # initialise randomforest
        X_train_df[str(pct_change)] = X_train_.values.reshape(-1,)
        X_test_df[str(pct_change)] = X_test_.values.reshape(-1,)
        # run over each column
        # fill first y values with 0 because pct_change is not able to do it
        # can only be done if i is not 0
        y_train_ = y_train.values
        y_test_ = y_test.values
        if(i > 0):
            y_train_ = np.insert(y_train_, 0, ([0 for p in range(i)]))[:-i]
            y_test_ = np.insert(y_test_, 0, ([0 for p in range(i)]))[:-i]

        if(i_end == (i+1) or (i % debug_i) == 0):
            # # XGBoost API example
            params = {'tree_method': 'gpu_hist', 'n_estimators': 100, 'objective':'binary:logistic'}
            dtrain = xgb.DMatrix(X_train_df.values, y_train_)
            dtest = xgb.DMatrix(X_test_df.values)
            model = xgb.train(params, dtrain, evals=[(dtrain, 'train')], verbose_eval=False)
            pred = model.predict(dtest)
            conf = confusion_matrix(y_test_, (pred>threshold_confusionm))
            print("treshhold: " + str(threshold_confusionm) + ' i: ' + str(i))
            print(conf)

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
