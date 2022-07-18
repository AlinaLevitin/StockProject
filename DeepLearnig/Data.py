from sklearn.model_selection import train_test_split


def scale_data(df):
    return (df - df.min()) / (df.max() - df.min())


class Data:

    def __init__(self, data):
        self.data = data
        self.x = self.data.iloc[:, :-3]
        self.y = self.data.iloc[:, -2:]


class TrainingData(Data):

    def __init__(self, data):
        super().__init__(data)
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.split_data()

    def split_data(self):
        x_train_and_val, x_test, y_train_and_val, y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                            random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train_and_val, y_train_and_val, test_size=0.2,
                                                          random_state=42)
        x_train = scale_data(x_train)
        x_val = scale_data(x_val)
        x_test = scale_data(x_test)

        return x_train, x_val, x_test, y_train, y_val, y_test
