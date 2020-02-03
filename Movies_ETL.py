#%%
# import dependencies
import json
import pandas as pd
import numpy as np
import re


from config import db_password
from sqlalchemy import create_engine
import psycopg2

import time


#%%

# go to the directory
file_dir = '/Users/francisrey/Desktop/Modules/Module_8/Movies_ETL/Resources'

# %%
# load the JSON file
with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)

#%%
# load movieMetadata and ratings from Kaggle
kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv')
ratings = pd.read_csv(f'{file_dir}/ratings.csv')

#%%
# count the records
len(wiki_movies_raw)
# 7311 records

# %%
# First 5 records
wiki_movies_raw[:5]

# %%
# Last 5 records
wiki_movies_raw[-5:]

#%%
# Some records in the middle
wiki_movies_raw[3600:3605]

#%%
# creating wiki_movies dataframe 
wiki_movies_df = pd.DataFrame(wiki_movies_raw)
wiki_movies_df.head()

#%%
# metadata dataframe
kaggle_metadata.head()

#%%
# ratings dataframe
ratings.head()

#%%
# column names in wiki_movies
wiki_movies_df.columns.tolist()

#%%
# identifying movies that has directors and imdb_link but not a series
wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]

#%%
len(wiki_movies)
# 7076 records

#%%
wiki_movies_df[wiki_movies_df['Arabic'].notnull()]

#%%
wiki_movies_df[wiki_movies_df['Arabic'].notnull()]['url']

#%%
# filtered dataframe (w director, w imdb, not a series)
wiki_df = pd.DataFrame(wiki_movies)
wiki_df.head()
#%%
# filtered list of columns
sorted(wiki_df.columns.tolist())

#len(sorted(wiki_df.columns.tolist()))
# 75 columns


#%%
#rerun copy
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie
#%%
# filetered dataframe using clean_movie()
clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)

#%%
# list of columns after clean_movie()
sorted(wiki_movies_df.columns.tolist())

#len(sorted(wiki_movies_df.columns.tolist()))
# 39 columns

#%%
# extract imdb_ID using Regex
wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
# 7076 records
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
# 7033 records
wiki_movies_df.head()


#%%
# count of null in each column
[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]

#len([[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns])
#40 columns

#%%
# dropping column if 90% of its content is null
[column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]

#len([column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9])
#21 columns

#%%
# 21 columns into a new wiki_movies_df
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

#%%
# double check the number of columns
len(sorted(wiki_movies_df.columns.tolist()))

#%%
#dropping data/records that didnt hit the box office
box_office = wiki_movies_df['Box office'].dropna() 

# count of box office data
#len(box_office)
#5485 records


#%%
# identifying columns that needs to be converted
wiki_movies_df.dtypes


#%%
# same expression using lambda syntax
lambda arguments: expression
lambda x: type(x) != str

#box_office[box_office.map(lambda x: type(x) != str)]

box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


#%%
# view datas that are not into string type
def is_not_a_string(x):
    return type(x) != str

box_office[box_office.map(is_not_a_string)]



#%%
# final forms

form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)

form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)

box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


#%%

box_office[~matches_form_one & ~matches_form_two]

#%%
# code to extract the data
box_office.str.extract(f'({form_one}|{form_two})')
# 5487 records

#%%
def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan


#%%
wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

#%%
#drop 'box office' column
wiki_movies_df.drop('Box office', axis=1, inplace=True)
wiki_movies_df.head()

#%%
# parse budget data Part 3
budget = wiki_movies_df['Budget'].dropna()

#%%
# convert list to string
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

#%%
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

#%%

matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
budget[~matches_form_one & ~matches_form_two]

#%%
# replace last line on code above w these:
budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]

#%%

wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

#%%
#drp the 'budget' column
wiki_movies_df.drop('Budget', axis=1, inplace=True)

#%%
# parse release date
release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

#%%
# parse
date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'

#%%
#extract
release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)
# 7001 rows
#%%
wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)
#wiki_movies_df.head()

#%%
# parse running time
running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

#%%
# most of the entries
running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()
#6528 rows
#%%
# the other entries
running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]

#%%
running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()
# 6877 entries

#%%

running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]

#%%

running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

#%%

running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

#%%
# hours into minutes
wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

#%%
# drop running time column
wiki_movies_df.drop('Running time', axis=1, inplace=True)

#%%
kaggle_metadata.dtypes

#%%
# check if value is true or false 
kaggle_metadata['adult'].value_counts()

#%%
# remove bad data
kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]

#%%
# deleting data rated as adult
kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

#%%
# values in video colomn
kaggle_metadata['video'].value_counts()

#%%
   
kaggle_metadata['video'] == 'True'
kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

#%%
kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

#%%
# date and time
ratings.info(null_counts=True)
pd.to_datetime(ratings['timestamp'], unit='s')
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

#%%

ratings['rating'].plot(kind='hist')
ratings['rating'].describe()

#%%
# merge dataframe (inner join)
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

#%%

# Competing data:
# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle
# running_time             runtime
# budget_wiki              budget_kaggle
# box_office               revenue
# release_date_wiki        release_date_kaggle
# Language                 original_language
# Production company(s)    production_companies  
#--------------------------------------------------------------------------

# title column (wiki and kaggle)
movies_df[['title_wiki','title_kaggle']]

#%%
# title columns that are different from one another ( wiki and kaggle)
movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]

#%%
# looking for null values
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]
# deciding to keep title_kaggle and drop title_wiki

#%%
# graphing a scatter plot
movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')

#%%
# comparing numerical values (budget_wiki / budget_kaggle) using a plot
movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')
# 
#%%
# scatter plot (box_office/revenue)
movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')

#%%
# scatter plot (revenue/box_office)
movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')

#%%
# plot
movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')

#%%

movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]

#%%

movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index

#%%
# dropping the row
movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)

#%%
# checking for null values (wiki) release date
movies_df[movies_df['release_date_wiki'].isnull()]

#%%
# checking for null values (kaggle) release date
movies_df[movies_df['release_date_kaggle'].isnull()]
# few missing data on wiki, none missing in kaggle: 
# will be dropping release_date_wiki.

#%%

#movies_df['Language'].value_counts()

# converting list into tuple
movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)

#%%

movies_df['original_language'].value_counts(dropna=False)

#%%

movies_df[['Production company(s)','production_companies']]
# kaggle is more consistent
# dropping wiki

#%%
# Wikipedia	              Kaggle	                 Resolution
# 1.title_wiki	          title_kaggle	         Drop Wikipedia.
# 2.running_time	      runtime	             Keep Kaggle; fill in zeros with Wikipedia data.
# 3.budget_wiki	          budget_kaggle	         Keep Kaggle; fill in zeros with Wikipedia data.
# 4.box_office	          revenue	             Keep Kaggle; fill in zeros with Wikipedia data.
# 5.release_date_wiki	  release_date_kaggle	 Drop Wikipedia.
# 6.Language	          original_language	     Drop Wikipedia.
# 7.Production company(s) production_companies	 Drop Wikipedia.

#%%
# this satisfies (1,5,6,7)
# dropping multiple colums
movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

#%%
# defining the replacement process

def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)


#%%
# calling the replacement process on (2,3,4)
fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df

#%%
# value_counts

for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)

#%%

movies_df['video'].value_counts(dropna=False)


#%%
# reordering columns
movies_df = movies_df[['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]

#%%
# renaming columns
movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)



#%%
# groupby userId and count. movieId as index
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                .rename({'userId':'count'}, axis=1) \
                .pivot(index='movieId',columns='rating', values='count')

#%%

rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

#%%

movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

#%%

movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

#%%
# load movies_df to postgress
db_string = f'postgres://postgres:{db_password}@127.0.0.1:5432/movie_data'
# sqlalchemy 
engine = create_engine(db_string)

# import movie data
movies_df.to_sql(name='movies', con=engine, if_exists='replace')

# %%
# Load rating csv to Postgres
rows_imported = 0
# get the start_time from time.time()
start_time = time.time()
for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done. {time.time() - start_time} total seconds elapsed')


#%%


