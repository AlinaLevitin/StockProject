import pandas as pd


def open_csv(file_name):
    # Reading the CSV:
    df = pd.read_csv(file_name, index_col=False)
    # Parsing the procId column:
    proc_id = df['procId'].str.split('_', expand=True)
    proc_id.columns = ['Symbol', 'Task-symbol', 'n', 'Date']
    # Removing the procId column from original table:
    df.drop(['procId'], axis=1, inplace=True)
    # Joining parsed procId and original data
    frames = [proc_id, df]
    new_df = pd.concat(frames, axis=1)
    return new_df


def get_winning_statistics(new_df, threshold=0.5, hits=1000, output='probability_to_win.csv'):
    # Total counts:
    new_df['Hits'] = 1
    total_statistics = new_df.groupby(['Task-symbol']).count()['Hits']

    # Winning counts:
    winning = pd.DataFrame(new_df[new_df['Balance'] > 0])
    winning['Win'] = 1
    winning_statistics = winning.groupby(['Task-symbol']).count()['Win']

    # Joining total and wining statistics
    statistics = [total_statistics, winning_statistics]
    final_df = pd.concat(statistics, axis=1)

    # Calculating winning Probability
    final_df['Probability to win'] = final_df['Win'] / final_df['Hits']

    # Slice selected data
    result = final_df.loc[(final_df['Probability to win'] > threshold) & (final_df['Hits'] > hits)]

    # Output data
    result.to_csv(output)

    # Print the results
    print(result)
