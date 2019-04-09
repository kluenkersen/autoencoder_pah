import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

class Analyse(object):
    """
        This class is tacking two files X and y for an random forest.
        Splits them and then starts training the random forest
        important to know it that while training the X file starts to getting
        stacked to create a time row. So X = X + new X+1
    """
    def __init__(self, filename_X, filename_y, filename, train_test_split=30,
                path_result='data/results', path='data', n_estimators=[10, 50, 100]):
        self.path = path
        self.path_result = path_result
        self.filename = filename
        # values of n_estimators for random forest
        self.n_estimators = n_estimators
        # class inizializer
        self.X_train_df = pd.DataFrame()
        self.X_test_df = pd.DataFrame()
        # read X and y from folders
        self.X = pd.read_csv(filename_X, header=None)
        self.y = pd.read_csv(filename_y)
        # define split range by %
        split = self.X.shape[0]/100*train_test_split
        self.split = int(split)
        # split train test
#         print(self.X)
        self.X_train = pd.DataFrame(self.X[0])[self.split:].reset_index(drop=True)
        self.y_train = self.y[self.split:].reset_index(drop=True)
        self.X_test = pd.DataFrame((self.X[0])[:self.split]).reset_index(drop=True)
        self.y_test = self.y[:self.split].reset_index(drop=True)
            
    def show_dates(self, y_test_, pred, threshold):
        df_t = pd.read_csv(self.path +'/'+ self.filename)
        df_t[:self.split].reset_index(drop=True)
        dates = []
        dates = df_t['Time (UTC)'].values[np.where(pred>threshold)[0]]
        print(dates)
        dt = ''
        for value in dates:
            if dt != value[:-9]:
                dt = value[:-9]
                print(value)
#         for value in dates:
#             print(value)

                    
    def fit(self, X_train, y_train, X_test, y_test, n_estimators=10, replace='012', shift='0', showdates=0):
        # replease just means which values are replaced, not shown values where gotten replaced by 0
        #train random forest
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        # predict result
        pred = clf.predict(X_test)
        #  create and save confusion matrix
        conf = confusion_matrix(y_test, pred)
        sfile = '/shift_' + shift + '_est_'+ str(n_estimators) +'_replace_' + replace + '_' + self.filename
        print(sfile)
        print(conf)
        np.save(self.path_result + sfile, conf)
        if(showdates == 1):
            self.show_dates(y_test_=y_test, pred=pred)
            
    def xgEvaluation(self, i_end=50, debug_i=5, threshold_confusionm=0.5, y_field='4', showdates=0):
        split = self.split
        X_train = self.X_train
        y_train = self.y_train[y_field]
        X_test = self.X_test
        y_test = self.y_test[y_field]
        y_train = pd.DataFrame(y_train).replace(to_replace=[2], value=1)
        y_test = pd.DataFrame(y_test).replace(to_replace=[2], value=1)

        y_train_ = pd.DataFrame()
        y_train_ = pd.DataFrame()
        X_train_df = pd.DataFrame()
        X_test_df = pd.DataFrame()
        
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
                if(showdates == 1):
                    self.show_dates(y_test_=y_test, pred=pred, threshold=threshold_confusionm)
        
    def analyse_result(self, showdates=0, i_start=1, i_end=50):
        # get columns for all y values
        y_train_ = pd.DataFrame()
        y_train_ = pd.DataFrame()
        y_columns = self.y.columns
        for i in range(i_end):
            pct_change = i * 5    
            if(pct_change == 0):
                pct_change = 1
            X_train_ = self.X_train.pct_change(periods=pct_change)
            X_test_ = self.X_test.pct_change(periods=pct_change)
            # get again right format for xgboost
            X_test_.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_train_.fillna(0, inplace=True)
            X_test_.fillna(0, inplace=True)
            X_train_ = pd.DataFrame(X_train_, dtype='float32')
            X_test_ = pd.DataFrame(X_test_, dtype='float32')
            # initialise randomforest
            self.X_train_df[str(pct_change)] = X_train_.values.reshape(-1,)
            self.X_test_df[str(pct_change)] = X_test_.values.reshape(-1,)
            # run over each column
            for column in self.y.columns:
                # fill first y values with 0 because pct_change is not able to do it
                # can only be done if i is not 0
                y_train_ = self.y_train[str(column)].values
                y_test_ = self.y_test[str(column)].values
                if(i > 0):
                    y_train_ = np.insert(y_train_, 0, ([0 for p in range(i)]))[:-i]
                    y_test_ = np.insert(y_test_, 0, ([0 for p in range(i)]))[:-i]
                for estimator in self.n_estimators:
                    # only for 0 and 1
#                     self.fit(X_train = self.X_train_df, 
#                         y_train = pd.DataFrame(y_train_).replace(to_replace=[2], value=0), 
#                         X_test = self.X_test_df, 
#                         y_test = pd.DataFrame(y_test_).replace(to_replace=[2], value=0), 
#                         n_estimators = estimator, 
#                         replace = '01_column-mulitplier_' + column,
#                         shift = str(i))
# #                     # only for 0 and 2
#                     self.fit(X_train = self.X_train_df, 
#                         y_train = pd.DataFrame(y_train_).replace(to_replace=[1], value=0), 
#                         X_test = self.X_test_df, 
#                         y_test = pd.DataFrame(y_test_).replace(to_replace=[1], value=0), 
#                         n_estimators = estimator, 
#                         replace = '02_column-mulitplier_' + column,
#                         shift = str(i))
                    # set 2 to 1 and get total overview
                    if i == i_start and column == '4':
                        self.fit(X_train = self.X_train_df, 
                            y_train = pd.DataFrame(y_train_).replace(to_replace=[2], value=1), 
                            X_test = self.X_test_df, 
                            y_test = pd.DataFrame(y_test_).replace(to_replace=[2], value=1), 
                            n_estimators = estimator, 
                            replace = '012_column-mulitplier_' + column,
                            shift = str(i), 
                            showdates = showdates)
    


