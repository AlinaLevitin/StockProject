import probability
import config


data = probability.Data("proc_table.csv")

data.get_winning_statistics("proc_table_probability_results.csv", config.hits, config.threshold)



