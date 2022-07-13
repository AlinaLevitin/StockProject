from csv import reader
import pandas as pd


class Data:

    def __init__(self, file_name):
        self.file_name = file_name
        self.data = self.open_csv()
        self.header = self.data.pop(0)
        self.data = self.split_name()

    def __repr__(self):
        print(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def open_csv(self):
        with open(self.file_name) as file:
            csv_reader = reader(file)
            return list(csv_reader)

    def split_name(self):
        for d in self.data:
            d[0] = d[0].split("_")
        return self.data

    def get_winning_statistics(self, output, hits, threshold):
        symbol = []
        date = []
        value = []

        for d in self.data:
            symbol.append(d[0][1])
            date.append(d[0][3])
            value.append(d[9])

        only_wins_symbol = []
        only_wins_date = []
        only_wins_value = []

        for d in self.data:
            if float(d[9]) > 0:
                only_wins_symbol.append(d[0][1])
                only_wins_date.append(d[0][3])
                only_wins_value.append(d[9])

        all_symbols = []
        total_count = []
        total_wins = []
        prob_win = []

        for s in symbol:
            if s not in all_symbols:
                x = symbol.count(s)
                y = only_wins_symbol.count(s)
                all_symbols.append(s)
                total_count.append(x)
                total_wins.append(y)
                prob_win.append(y / x)

        dataset = {'symbol': all_symbols, 'hits': total_count, 'wins': total_wins, 'probability to win': prob_win}

        df = pd.DataFrame(dataset)
        significant = df[df['hits'] > hits]

        result = significant[significant['probability to win'] > threshold]
        result.to_csv(path_or_buf=output, index=False)
