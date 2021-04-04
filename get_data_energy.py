import twint
import pandas as pd
import os

def scrape_tweets(keyword):
    c = twint.Config()
    c.Search = str(keyword)
    c.Limit = 10000
    c.Pandas = True
    c.Hide_output = True
    
    twint.run.Search(c)
    
    df = twint.storage.panda.Tweets_df
    
    df = df.drop(columns=['id', 'conversation_id', 'created_at', 'link', 'thumbnail', 'quote_url'])
    df = df[df['language'] == 'en']
    
    return df

# Scrape tweets related to coal, solar, wind, natural gas, petroleum
df_coal = scrape_tweets('coal energy')
df_solar = scrape_tweets('solar energy')
df_wind = scrape_tweets('wind energy')
df_gas = scrape_tweets('natural gas energy')
df_petro = scrape_tweets('petro energy')

# Paths to save pickles
path = os.getcwd()
path_coal = os.path.join(path, 'data', 'coal.pkl')
path_solar = os.path.join(path, 'data', 'solar.pkl')
path_wind = os.path.join(path, 'data', 'wind.pkl')
path_gas = os.path.join(path, 'data', 'gas.pkl')
path_petro = os.path.join(path, 'data', 'petro.pkl')

# Save pickles of dataframes
df_coal.to_pickle(path_coal)
df_solar.to_pickle(path_solar)
df_wind.to_pickle(path_wind)
df_gas.to_pickle(path_gas)
df_petro.to_pickle(path_petro)

print('All done!')
