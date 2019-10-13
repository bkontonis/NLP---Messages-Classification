import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the specified datasets
    
    Input:
        messages_filepath: Path to the messages data file
        categories_filepath: Path to the categories data file
    Output:
        df: dataframe with messages and categoried data merged
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    
    return df


def clean_data(df):
    """Cleans the data
    
    Input:
        df: dataframe with messages and categories information
    Output:
        df: the dataframe without duplicates and categories encoded
    """
    categories = df.categories.str.split(';', expand=True)
    category_colnames = categories.iloc[0].apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].astype(int)
    
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """Saves the cleaned data in a database
    
    Input:
        df: the cleaned dataframe
        database_filename: the name of the database
    Output:
        None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    sql = "DROP TABLE IF EXISTS LabeledMessages"
    engine.execute(sql)
    df.to_sql('LabeledMessages', engine, index=False)


def main():
    """Executes all the steps required for the data process:
           - Extract the data from the csv files
           - Merges and Cleans the data
           - Load the final cleaned dataset in a database
           
       Input: None
       Output: When successfully runs a database object will be created in your working directory
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()