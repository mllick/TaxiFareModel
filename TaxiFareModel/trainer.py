from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.data import get_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.data import clean_data
import pandas as pd



class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.X = X
        self.y = y
        self.pipeline =None


    # implement set_pipeline() function
    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                            ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                        remainder="drop")
        self.pipeline = Pipeline([('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())])
        #return self.pipe

    # implement train(self) function
    # def train(self):
    #     '''returns a trained pipelined model'''
    #     self.pipeline.fit(self.X, self.y)
    #     return self.pipeline

    # implement evaluate() function
    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        # store the data in a DataFrame


        # df = get_data()

        # # hold out
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

        # # build pipeline
        # pipeline = self.set_pipeline()

        # # train the pipeline
        # self.train(X_train, y_train, pipeline)

        # evaluate the pipeline
        #rmse = evaluate(X_val, y_val, self.pipeline)

if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df['fare_amount']
    X = df.drop("fare_amount",axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train
    Pipe_train = Trainer(X,y)
    Pipe_train.run()
    # evaluate
    metric = Pipe_train.evaluate(X_val, y_val)
    print(metric)
