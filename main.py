import probability
import config

df = probability.open_csv('proc_table.csv')
probability.get_winning_statistics(df, config.threshold, config.hits)



