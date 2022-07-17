import pickle


def save_pickle(data):
    pickle.dump(data, open("data.p", "wb"))


