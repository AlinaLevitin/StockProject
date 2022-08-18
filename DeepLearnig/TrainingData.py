from .Data import Data

from sklearn.model_selection import train_test_split


class TrainingData(Data):
    # A class to collect data for deeplearning
    def __init__(self, cwd):
        super().__init__(cwd)
        self.data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.train_num = None
        self.val_num = None
        self.test_num = None

    def __repr__(self):
        return f"training data: {self.train_num}, validation data: {self.val_num}, test data: {self.test_num}"

    def split_data(self):
        self.x = self.data.iloc[:, :-3]
        self.y = self.data.iloc[:, -2:]

        x_train_and_val, x_test, y_train_and_val, y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                            random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train_and_val, y_train_and_val, test_size=0.2,
                                                          random_state=42)
        x_train = self.scale_data(x_train)
        x_val = self.scale_data(x_val)
        x_test = self.scale_data(x_test)

        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_num = self.x_train.shape[0]
        self.val_num = self.x_val.shape[0]
        self.test_num = self.x_test.shape[0]

    @staticmethod
    def scale_data(df):
        return (df - df.min()) / (df.max() - df.min())

