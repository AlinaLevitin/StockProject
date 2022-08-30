from .Data import Data

from sklearn.model_selection import train_test_split


class TrainingData(Data):
    """

    A class to collect data for deep-learning training \n
    Use the class function split_data to split the data to train, validation and test data sets

    ...

    Attributes
    ----------
    cwd : str
        a string of the working directory
    steps_back : int
        the number of time points in the past
    steps_forward : int
        the number of time points in the future
    percent : int
        percent as the threshold for win or loss outcome
    interval : int
        interval between time points collected
    data : pandas DataFrame
        all the data collected according to the above parameters
        use class method read_all_data with the chosen parameters
        or open existing data with class method open_all_data

    Methods
    -------
    FROM Data CLASS:
    save_all_data(self, name='data')
        saves the data as csv file and the parameters as json file
        default name is 'data'
    open_all_data(self, name='data')
        opens the data from csv file and the parameters from json file
        default name is 'data'
    read_all_data(self, steps_back, steps_forward, percent, interval)
        reads the data according to the chosen parameters
    read_and_get_values(self, file)
        reads the data according to the chosen parameters from a chosen file
    split_data(self):
        splits the data to train, validation and test data sets randomly

    """

    def __init__(self, cwd: str):
        """

        :param cwd: a string of the working directory

        """

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
        """splits the data to train, validation and test data sets randomly"""
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
        """
        to scale the data to range between -1 and 1

        :param df: pandas DataFrame of inputs
        :return: pandas DataFrame with data ranging between -1 and 1
        """

        return (df - df.min()) / (df.max() - df.min())

