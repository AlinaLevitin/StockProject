import probability
import config
import os
import read_data_for_DL

cwd = os.getcwd()

# df = probability.open_csv('proc_table.csv')
# probability.get_winning_statistics(df, config.threshold, config.hits)

read_data_for_DL.read_all_data(cwd)



