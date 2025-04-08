################################################################################
# Matthew Spitulnik ############################################################
# Deep Learning in Practice ####################################################
# Stock Price Estimator ########################################################
# Project Summary: The goal of this project was to build a deep learning model with an embedded layer that could use the text from news articles about stocks to predict if the respective stock price would go up or down during the following market opening. To do this, I defined my own interactive function that allowed a user to input a list of company names, then used a text mining script to download news articles related to each stock name from investing.com and organized it all into a data frame. I created another function that took the resulting stock news article data frame, and utilized the yahoo finance API to collect the opening price of each stock the morning the article was released and the price of the when the market next opened. I then used various deep learning techniques, focused around embedded RNNs, to see if all of the collected data could be used to build models that would correctly predict if the stock price have gone up or down by the next market open.
################################################################################

################################################################################
### Install and load required packages #########################################
################################################################################
"""

#Install required packages
#%pip install yfinance
#%pip install datetime
#%pip install pandas
#%pip install requests
#%pip install lxml
#%pip install bs4
#%pip install regex
#%pip install IPython
#%pip install contractions
#%pip install nltk
#%pip install inflect
#%pip install gensim
#%pip install wget
#%pip install patool
#%pip install tensorflow tensorflow-hub
#%pip install fasttext-wheel
#%pip install seaborn
#%pip install matplotlib

#Import required packages
import yfinance as yf
import pandas as pd
import requests
from lxml import html
from bs4 import BeautifulSoup
import re
import os
from IPython.display import HTML
import time
import datetime as dt
from datetime import datetime, timedelta
import sys
import random
from statistics import mean
from sklearn.model_selection import train_test_split
import patoolib
import numpy as np
import gzip
import shutil

import tensorflow
from tensorflow import keras
from tensorflow.python.keras.layers import Layer
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import layers
from keras.layers import Embedding,Conv1D, MaxPooling1D, Lambda
import tensorflow_hub as hub
import keras.backend as K
from keras.optimizers import Adam
from keras import regularizers

import contractions
import nltk
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
import re
import inflect
wordPlur=inflect.engine()
import gensim
from gensim.parsing.preprocessing import remove_stopwords
gensim_stopwords = gensim.parsing.preprocessing.STOPWORDS
from gensim.models import FastText
import fasttext
import fasttext.util
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

"""################################################################################
### Import the data sets that will be used throughout the code #################
################################################################################
"""

###This creates a "default_directory" variable, where the directory path to the
# data files folder containing all of the required data sets is saved so that it
# does not need to be constantly re-entered. Remember to use forward slashes
# instead of back slashes in the directory path. For example, if the datafiles
# folder is saved in "C:\home\project\datafiles", then "C:/home/project"
# would be inserted between the quotes below.
default_directory = "<DEFAULT DIRECTORY PATH HERE>"

#now import the files that will be needed throughout the code
art_list_full=pd.read_csv(f'{default_directory}/datafiles/art_list_full.csv')
art_list_full2=pd.read_excel(f'{default_directory}/datafiles/art_list_full_v2.xlsx')
stock_price_DF=pd.read_csv(f'{default_directory}/datafiles/stock_price_11_26.csv')
art_price_DF=pd.read_csv(f'{default_directory}/datafiles/art_price_DF.csv')

"""################################################################################
### Create the functions that can be used for collecting news articles about stocks
################################################################################
"""

# First create the interactive function that allows a user to input a stock name and how many articles they would like about that stock
def stock_news_collect():
    stock=input("Enter the name or ticker of the stock you would like to collect news articles about: ")

    tmp_query=f'https://www.investing.com/search/?q={stock}'

    tempqueryhtml=requests.get(tmp_query)
    query_page=BeautifulSoup(tempqueryhtml.text,'html.parser')
    query_text=query_page.find(class_='js-inner-all-results-quotes-wrapper newResultsContainer quatesTable')
    query_table=query_text.find_all(class_='js-inner-all-results-quote-item row')

    country, ticker, full_name, exchange, link = [],[],[],[],[]
    for i in range(0,len(query_table)):
        tmp_country=re.findall('(?<=ceFlags middle )(.*)(?="></i></span>)',str(query_table[i]))
        country.append(tmp_country[0])

        tmp_tick=re.findall('(?<= class="second">)(.*)(?=</span>)',str(query_table[i]))
        ticker.append(tmp_tick[0])

        tmp_name=re.findall('(?<= class="third">)(.*)(?=</span>)',str(query_table[i]))
        if "&amp;" in tmp_name[0]:
          tmp_name[0]=re.sub('&amp;','&',tmp_name[0])
        full_name.append(tmp_name[0])

        tmp_ex=re.findall('(?<= class="fourth">)(.*)(?=</span>)',str(query_table[i]))
        exchange.append(tmp_ex[0])

        part_link=re.findall('(?<=href=")(.*)(?=">)',str(query_table[i]))
        full_link="https://www.investing.com" + part_link[0]
        link.append(full_link)

    tmp_stock_df=pd.DataFrame()
    tmp_stock_df['country']=country
    tmp_stock_df['ticker']=ticker
    tmp_stock_df['full_name']=full_name
    tmp_stock_df['exchange']=exchange
    tmp_stock_df['link']=link

    display(HTML(tmp_stock_df.to_html(render_links=True, escape=False)))

    selection=input(f"Confirm the number index of the stock you wanted news articles for (0-{max(tmp_stock_df.index)}): ")

    stock_link=tmp_stock_df.loc[int(selection),'link']+'-news'

    tempstockhtml=requests.get(stock_link)
    stock_page=BeautifulSoup(tempstockhtml.text,'html.parser')

    stock_page_index=stock_page.find_all(class_='flex gap-2 items-center')
    stock_page_count=stock_page_index[0].find_all(class_='flex rounded leading-5 border font-semibold items-center border-[#F7F7F8] bg-[#F7F7F8] text-[#1256A0] p-[11px]')
    page_count_len=int(len(stock_page_count))-1
    total_page_count=int(stock_page_count[page_count_len].text)

    pages=input(f'How many pages worth of articles would you like to collect (1-{total_page_count}):')

    full_news_DF=pd.DataFrame()
    for pc in range(0,int(pages)):
      pages_1=pc+1
      stock_link=tmp_stock_df.loc[int(selection),'link']+'-news/'+str(pages_1)
      tempstockhtml=requests.get(stock_link)
      stock_page=BeautifulSoup(tempstockhtml.text,'html.parser')
      stock_text=stock_page.find_all(class_='block w-full sm:flex-1')

      art_name, art_source, art_date, art_link, art_text = [],[],[],[],[]
      for i in range(0,len(stock_text)):
        if 'mt-2.5' not in str(stock_text[i]):
          temp_name=stock_text[i].find(class_='inline-block text-sm leading-5 sm:text-base sm:leading-6 md:text-lg md:leading-7 font-bold mb-2 hover:underline')
          art_name.append(temp_name.text)

          tmp_source=re.findall('(?<="news-provider-name">)(.*)(?=</span></li>)',str(stock_text[i]))
          art_source.append(tmp_source[0])

          temp_link=stock_text[i].find(class_='inline-block text-sm leading-5 sm:text-base sm:leading-6 md:text-lg md:leading-7 font-bold mb-2 hover:underline')
          part_link=re.findall('(?<=href=")(.*)(?=">)',str(temp_link))
          full_link="https://www.investing.com" + part_link[0]
          art_link.append(full_link)

          tempnewshtml=requests.get(full_link)
          news_page=BeautifulSoup(tempnewshtml.text,'html.parser')

          date_find=news_page.find_all(class_='contentSectionDetails')
          for date in range(0,len(date_find)):
            if "Published" in date_find[date].text:
              temp_date=re.findall('(?<=Published )(.*)(?=</span>)',str(date_find[date]))
              temp_date=temp_date[0].replace(' ET','')
              datetime_object = datetime.strptime(temp_date, '%b %d, %Y %I:%M%p')
          art_date.append(datetime_object)

          news_body=news_page.find(class_='WYSIWYG articlePage')
          news_text=news_body.find_all('p')
          tmp_str=str()
          for i in range(0,len(news_text)):
            if "Position added successfully to:" not in news_text[i].text:
              tmp_str=tmp_str+' ' + news_text[i].text
          art_text.append(tmp_str)
          print('Collected the text for an article. Now pausing for 15 seconds...')
          time.sleep(15)

      tmp_news_df=pd.DataFrame()
      tmp_news_df['art_name']=art_name
      tmp_news_df['art_source']=art_source
      tmp_news_df['art_date']=art_date
      tmp_news_df['art_link']=art_link
      tmp_news_df['art_text']=art_text

      temp_comb_DF=[full_news_DF,tmp_news_df]
      full_news_DF=pd.concat(temp_comb_DF)

      display(f"Just finished page {pages_1}.")

    full_news_DF=full_news_DF.reset_index(drop=True)

    return HTML(full_news_DF.to_html(render_links=True, escape=False))

#run the function and save the output to a file. "stock_news" can be any name you'd like.
stock_news=stock_news_collect()

#take a look at the info
stock_news

### Now Create the function for looking up articles for multiple stocks at once: stock_news_collect_list(stock_list,pages).
# "stock_list" input should be a list of stock names or tickers.
# "pages" input should be how many pages worth of articles you'd like to collect for each stock.
# 2 objects are outputted:
# 1) A dataframe with all of the collected news articles
# 2) A list of all the stocks that articles could not be collected for
def stock_news_collect_list(stock_list,pages):
  full_news_DF=pd.DataFrame()
  fail_list=[]
  for stock in stock_list:
    try:
      tmp_query=f'https://www.investing.com/search/?q={stock}'
      tempqueryhtml=requests.get(tmp_query)
      query_page=BeautifulSoup(tempqueryhtml.text,'html.parser')
      query_text=query_page.find(class_='js-inner-all-results-quotes-wrapper newResultsContainer quatesTable')
      query_table=query_text.find_all(class_='js-inner-all-results-quote-item row')
      part_link=re.findall('(?<=href=")(.*)(?=">)',str(query_table[0]))
      stock_link="https://www.investing.com" + part_link[0] + '-news/'

      tmp_country=re.findall('(?<=ceFlags middle )(.*)(?="></i></span>)',str(query_table[0]))
      tmp_tick=re.findall('(?<= class="second">)(.*)(?=</span>)',str(query_table[0]))
      tmp_name=re.findall('(?<= class="third">)(.*)(?=</span>)',str(query_table[0]))
      if "&amp;" in tmp_name[0]:
        tmp_name[0]=re.sub('&amp;','&',tmp_name[0])
      tmp_ex=re.findall('(?<= class="fourth">)(.*)(?=</span>)',str(query_table[0]))

      tempstockhtml=requests.get(stock_link)
      stock_page=BeautifulSoup(tempstockhtml.text,'html.parser')

      stock_page_index=stock_page.find_all(class_='flex gap-2 items-center')
      stock_page_count=stock_page_index[0].find_all(class_='flex rounded leading-5 border font-semibold items-center border-[#F7F7F8] bg-[#F7F7F8] text-[#1256A0] p-[11px]')
      page_count_len=int(len(stock_page_count))-1
      total_page_count=int(stock_page_count[page_count_len].text)

      if pages > total_page_count:
        pages=total_page_count

      for pc in range(0,int(pages)):
        pages_1=pc+1
        tmp_stock_link=stock_link
        tmp_stock_link=tmp_stock_link+str(pages_1)
        tempstockhtml=requests.get(tmp_stock_link)
        stock_page=BeautifulSoup(tempstockhtml.text,'html.parser')
        stock_text=stock_page.find_all(class_='block w-full sm:flex-1')

        stock_name,stock_tick,stock_country,stock_ex, art_name, art_source, art_date, art_link, art_text = [],[],[],[],[],[],[],[],[]
        for i in range(0,len(stock_text)):
          if 'mt-2.5' not in str(stock_text[i]):
            stock_name.append(tmp_name[0])
            stock_tick.append(tmp_tick[0])
            stock_country.append(tmp_country[0])
            stock_ex.append(tmp_ex[0])

            temp_name=stock_text[i].find(class_='inline-block text-sm leading-5 sm:text-base sm:leading-6 md:text-lg md:leading-7 font-bold mb-2 hover:underline')
            art_name.append(temp_name.text)

            tmp_source=re.findall('(?<="news-provider-name">)(.*)(?=</span></li>)',str(stock_text[i]))
            art_source.append(tmp_source[0])

            temp_link=stock_text[i].find(class_='inline-block text-sm leading-5 sm:text-base sm:leading-6 md:text-lg md:leading-7 font-bold mb-2 hover:underline')
            part_link=re.findall('(?<=href=")(.*)(?=">)',str(temp_link))
            full_link="https://www.investing.com" + part_link[0]
            art_link.append(full_link)

            tempnewshtml=requests.get(full_link)
            news_page=BeautifulSoup(tempnewshtml.text,'html.parser')

            date_find=news_page.find_all(class_='contentSectionDetails')
            for date in range(0,len(date_find)):
              if "Published" in date_find[date].text:
                temp_date=re.findall('(?<=Published )(.*)(?=</span>)',str(date_find[date]))
                temp_date=temp_date[0].replace(' ET','')
                datetime_object = datetime.strptime(temp_date, '%b %d, %Y %I:%M%p')
            art_date.append(datetime_object)

            news_body=news_page.find(class_='WYSIWYG articlePage')
            news_text=news_body.find_all('p')
            tmp_str=str()
            for i in range(0,len(news_text)):
              if "Position added successfully to:" not in news_text[i].text:
                tmp_str=tmp_str+' ' + news_text[i].text
            art_text.append(tmp_str)
            print(f'Collected the text for an article about {stock}. Now pausing for 15 seconds...')
            time.sleep(15)

        tmp_news_df=pd.DataFrame()
        tmp_news_df['stock_name']=stock_name
        tmp_news_df['stock_tick']=stock_tick
        tmp_news_df['stock_country']=stock_country
        tmp_news_df['stock_ex']=stock_ex
        tmp_news_df['art_name']=art_name
        tmp_news_df['art_source']=art_source
        tmp_news_df['art_date']=art_date
        tmp_news_df['art_link']=art_link
        tmp_news_df['art_text']=art_text

        temp_comb_DF=[full_news_DF,tmp_news_df]
        full_news_DF=pd.concat(temp_comb_DF)

        display(f"Just finished {stock}, page {pages_1}.")

        full_news_DF=full_news_DF.reset_index(drop=True)
        ###If your funciton keeps timing out, uncomment this below line and a CSV file will be exported each time it iterates, allowing you to progressively save your work.
        #full_news_DF.to_csv(f'{default_directory}/datafiles/full_news_DF.csv',header=True,index=False)
    except AttributeError:
      print(f'Could not locate a page for {stock}')
      fail_list.append(stock)
      continue
  return full_news_DF, fail_list

#Now test the function
stock_news_collect_list(['apple','EA','Tesla'],3)

#A list of trending stocks was manually downloaded from web.stockedge.com, then copied into a text file.
# The text file will now be imported and converted into a list so that it can be used in the function that
# collects news articles for a list of stocks.
file = open(f'{default_directory}/datafiles/trending stock list.txt', 'r')
trending_stock_list = file.read()
trending_stock_list = trending_stock_list.split('\n')
trending_stock_list= [i for i in trending_stock_list if i]
file.close()
trending_stock_list

#See if there are any duplicate stocks
print(len(trending_stock_list))
print(len(set(trending_stock_list)))

#Some stocks are listed multiple times, so reduce duplicates
final_stock_list=set(trending_stock_list)
final_stock_list=list(final_stock_list)
len(final_stock_list)

final_stock_list

#Export the file so it can be easily imported as needed
with open(f'{default_directory}/datafiles/final stock list.txt', 'w') as file:
    for i in final_stock_list:
        file.write(str(i) + '\n')

file.close()

#import the final trending stock list if needed
file = open(f'{default_directory}/datafiles/final stock list.txt', 'r')
final_stock_list = file.read()
final_stock_list = final_stock_list.split('\n')
final_stock_list= [i for i in final_stock_list if i]
file.close()
final_stock_list

###Now use the trending stock list in the function that collects news articles based on a list of stocks.
#IMPORTANT NOTE# This code took multiple iterations to run, over a span of multiple days,
# so once it finished I exported the final CSV file so that they can just be imported going
# forward. There is code to import it at the top of the page already, and the file is art_list_full.

#art_list_full, fail_list=stock_news_collect_list(final_stock_list,5)

#Export the final list of stock articles
#art_list_full.to_csv(f'{default_directory}/datafiles/art_list_full.csv',header=True,index=False)

###It was determined the amount of text in some of the articles was making it difficult to properly export and import the data.
# It was also causing some blank rows to appear. To compensate for this, the blank rows were manually removed and the file was
# converted to an excel document instead. It was then imported below. The file will also be imported in the beginning of the code.

#art_list_full2=pd.read_excel(f'{default_directory}/datafiles/art_list_full_v2.xlsx')

#Because CSVs and excel docs can only contain a certain amount of characters per cell, the text from some articles was being stretched
# across multiple unnamed columns. The below code ensures all of the text is combined into one column, and then removes all unneeded columns.

#first combine the text for each article into one set of text, then add each article's text as an item in a list
comb_arts=[]
for i in art_list_full2.index:
    temp_str=''
    for h in art_list_full2.columns[8:]:
      if not pd.isna(art_list_full2.loc[i,h]):
        temp_str=temp_str+str(art_list_full2.loc[i,h])
    comb_arts.append(temp_str)

#now set the main art_text column to be the combined text for each article
art_list_full2['art_text']=comb_arts
#now remove the additional columns
art_list_full2=art_list_full2.iloc[:,:9]

"""################################################################################
### Create the function for collecting stock prices based on article date ######
################################################################################
"""

#Now create the function that takes in a DF of news articles, and gets
# the opening price of the stock each article is about over a number of
# days equal to input_days, starting from when the article came out.
def stock_price_collection(input_df,input_days):
  stock_price_DF=pd.DataFrame()
  perm_ticker='blank_to_start'
  for i in input_df.index:
    tmp_tick=input_df.loc[i,'stock_tick']
    start_date=pd.to_datetime(input_df.loc[i,'art_date']).tz_localize('US/Eastern',ambiguous=True)
    end_date=start_date+dt.timedelta(days=input_days)
    end_date=end_date.tz_convert('US/Eastern')
    stock_price_DF.loc[i,'stock_tick']=tmp_tick
    stock_price_DF.loc[i,'art_date']=start_date

    if perm_ticker != tmp_tick:
      perm_ticker = tmp_tick
      tempStockInfo=yf.Ticker(str(tmp_tick))

    try:
      saved_info=tempStockInfo.history(start=start_date, end=end_date)
    except:
      end_date=end_date +dt.timedelta(hours=1)
      saved_info=tempStockInfo.history(start=start_date, end=end_date)

    for d in range(0,len(saved_info)):
      stock_price_DF.loc[i,f'day_{d}_open']=saved_info.loc[saved_info.index[d],'Open']
      stock_price_DF.loc[i,f'day_{d}_close']=saved_info.loc[saved_info.index[d],'Close']

  return stock_price_DF

###Now collect the price data based on the article dates
#IMPORTANT NOTE# This code took a while to run, so once it finished I
# exported the CSV file so that it can just be imported going forward.
# There is code to import it at the top of the page already, file is stock_price_11_26

#stock_price_DF=stock_price_collection(art_list_full2,3)

stock_price_DF

#Export the DF so that it can be imported as needed later.

#stock_price_DF.to_csv(f'{default_directory}/datafiles/stock_price_11_26.csv',header=True,index=False)

#Now merge the data together into one data frame, but only keep the columns that will be important for the modeling.
art_price_DF = pd.merge(art_list_full2[['stock_name','stock_tick','art_name','art_date','art_text']],
                        stock_price_DF[['day_0_open','day_1_open']],how='left',left_index=True, right_index=True)

art_price_DF

#Some articles may have come out on days when the market was closed, or the market was closed on the following day,
# in which case there would be no day 0 or day 1 open price, so those rows will be removed.
# Some articles also did not actually contain any text for whatever reason, so those rows will also be removed.

art_price_DF=art_price_DF[art_price_DF['day_1_open'].notnull()]
art_price_DF=art_price_DF[art_price_DF['art_text'] != '']

#Now create a column that establishes if the price of the stock increased, decreased, or stayed the same from day 0 to day 1.
for i in art_price_DF.index:
  if art_price_DF.loc[i,'day_0_open'] > art_price_DF.loc[i,'day_1_open']:
    art_price_DF.loc[i,'increase']='no'
  elif art_price_DF.loc[i,'day_0_open'] < art_price_DF.loc[i,'day_1_open']:
    art_price_DF.loc[i,'increase']='yes'
  else:
    art_price_DF.loc[i,'increase']='same'

art_price_DF

#Reset the index of the DF
art_price_DF=art_price_DF.reset_index(drop=True)

###Whenever the merged dataframe is exported and inported, we will still face the issue of the text from the articles stretching across multiple columns.
# This below process will write the text to individual text documents so that it is all contained in one entry, then each article's text will be imported
# indvidually, added as an item in a list, then added back to the dataframe.

#First write the text to individual text files
for i in art_price_DF.index:
  with open(f'{default_directory}/datafiles/text/art_{i}.txt', 'w', errors='ignore') as file:
    file.write(str(art_price_DF.loc[i,'art_text']))
    file.close()

#Then import the text back in as individual text files
final_text_info = []
for i in range(0,len(art_price_DF)):
  with open(f'{default_directory}/datafiles/text/art_{i}.txt') as file:
    final_text_info.append(file.read().strip('\n'))
    file.close()

#Now drop the text column from the dataframe
art_price_DF=art_price_DF.drop('art_text',axis=1)

#Now export and import the file back in (this file is also imported at the beginning of the code)
#art_price_DF.to_csv(f'{default_directory}/datafiles/art_price_DF.csv',header=True,index=False)
#art_price_DF=pd.read_csv(f'{default_directory}/datafiles/art_price_DF.csv')

#Now add the text data back in
art_price_DF['art_text']=final_text_info

#Finally, factorize the "increase" column
art_price_DF['inc_factor'] = pd.factorize(art_price_DF['increase'])[0]

art_price_DF

"""################################################################################
### Now build the deep learning models that will try to predict if the stock will go up or down based on the text from the news articles
################################################################################
"""

###first create the tokenized words
#tokenizer = Tokenizer(num_words=20000) <-----using a max terms of 20000, one epoch would hae taken almost 5 hours
tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(art_price_DF['art_text'])
sequences = tokenizer.texts_to_sequences(art_price_DF['art_text'])
tok_art_text = pad_sequences(sequences)

tok_art_text

len(tok_art_text)

#Create the training and test data
X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_DF['inc_factor'], test_size=0.3)

###Now start with my base deep learning model: embedded layer, bidirectional LSTM, dropout layer, dense layer, activation layer
# model 1
'''
inputs = keras.Input(shape = (None,), dtype = 'int64')

#embedded = layers.Embedding(input_dim = 5000,output_dim = 256)(inputs) <-----after bringing input dims down to 5000, one epoch was still going to take almost 4 hours, adjusting output dim now
embedded = layers.Embedding(input_dim = 5000,output_dim = 32)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')
'''
#top acc: .8085
#top val: .5989
#test acc: .59

#decrease input, decrease output, decrease LSTM
# model 2
'''
inputs = keras.Input(shape = (None,), dtype = 'int64')

#embedded = layers.Embedding(input_dim = 5000,output_dim = 256)(inputs) <-----after bringing input dims down to 5000, one epoch was still going to take almost 4 hours, adjusting output dim now
#embedded = layers.Embedding(input_dim = 5000,output_dim = 32,mask_zero = True)(inputs)<-----still taking a few hours
#embedded = layers.Embedding(input_dim = 100,output_dim = 32,mask_zero = True)(inputs) #<-----still taking about 40 minutes,going to try reducing LSTM layers
#embedded = layers.Embedding(input_dim = 100,output_dim = 16,mask_zero = True)(inputs) #<-----still taking about 40 minutes after reducing LSTM layer, going to try reducing output_dim now
embedded = layers.Embedding(input_dim = 500,output_dim = 8,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
#x = layers.Bidirectional(layers.LSTM(32))(embedded) <----after reducing embedding layer input a few times, epoch time was still about 40 minutes each, trying to reduce this now
x = layers.Bidirectional(layers.LSTM(8))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')
'''

#Increased input dim, output dim, lstm
# model 3

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(art_price_DF['art_text'])
sequences = tokenizer.texts_to_sequences(art_price_DF['art_text'])
tok_art_text = pad_sequences(sequences)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

#embedded = layers.Embedding(input_dim = 5000,output_dim = 256)(inputs) <-----after bringing input dims down to 5000, one epoch was still going to take almost 4 hours, adjusting output dim now
#embedded = layers.Embedding(input_dim = 5000,output_dim = 32)(inputs) <------took about an hour 15 per epoch, trying to drop that down a bit
embedded = layers.Embedding(input_dim = 1000,output_dim = 32)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#Increased input dim, added embedding mask
####This one was going to take too long
# model 4

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(art_price_DF['art_text'])
sequences = tokenizer.texts_to_sequences(art_price_DF['art_text'])
tok_art_text = pad_sequences(sequences)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

#embedded = layers.Embedding(input_dim = 5000,output_dim = 256)(inputs) <-----after bringing input dims down to 5000, one epoch was still going to take almost 4 hours, adjusting output dim now
embedded = layers.Embedding(input_dim = 5000,output_dim = 32,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

"""#Now look at cleaning the text data to see if it makes a difference with the results"""

#create the code for cleaning the data
cln_txt_DF=pd.DataFrame()
for i in art_price_DF.index:
  #make everything lowercase and fix contractions
  tmpText=contractions.fix(art_price_DF.loc[i,'art_text'].lower())
  #remove the players name if it appears in the review
  tmpText=re.sub(str(art_price_DF.loc[i,'stock_tick']).lower(),'',tmpText)
  tmpText=re.sub(str(art_price_DF.loc[i,'stock_name']).lower(),'',tmpText)
  # Remove any kind of hyphen-like character at the beginning
  tmpText = re.sub(r"^[\u002D\u2010\u2011\u2012\u2013\u2014\u2015]+", "", tmpText)
  #remove words that contain numbers/other characters
  tempTxt = re.sub(r"[^A-Za-z\-]", " ", tmpText).split()
  #lemmetize and make everything singular
  tempRevTags=nltk.pos_tag(tempTxt)
  tempRevNew=[]
  for ct in tempRevTags:
    tagList=re.match(r'V',ct[1]) or re.match(r'JJ',ct[1])
    if tagList:
      tempRevNew.append(lem.lemmatize(ct[0],'v'))
    else:
      tempRevNew.append(ct[0])
  corrected_words = []
  for w in tempRevNew:
    w = w.lstrip('-')  # Remove hyphens from the beginning
    if w:
      if wordPlur.singular_noun(w) is False:
        corrected_words.append(w)
      else:
        singular_word = wordPlur.singular_noun(w)
        corrected_words.append(singular_word)

  tempRevNew = corrected_words  # Update with corrected words
  #remove stop words, or words that are 3 characters or less, or more then 13 characters
  tempTxtNew=[word for word in tempRevNew if word not in gensim_stopwords and len(word)>3 and len(word)<=13]
  cln_txt_DF.loc[i,'text']=' '.join(r for r in tempTxtNew)

#now look at some quick stats related to the cleaned up text
#First get the length (characters) of each article and add it to a list
length_cln_text=[]
for i in cln_txt_DF['text']:
  length_cln_text.append(len(i))

#get the average length (characters) of each article
print(mean(length_cln_text)) #1450.90881006865
#max length
print(max(length_cln_text)) #39961
#min length
print(min(length_cln_text)) #0

#see how many articles have zero length, or no text
cln_txt_DF[cln_txt_DF['text']==''] #4049
for i in cln_txt_DF.index:
  if len(cln_txt_DF.loc[i,'text'])==0:
    print(i) #only article with index  has a length of 0 4049

cln_txt_DF

#Now once again, export the txt to individual text files, since we will have
# to deal with the text being stretched across multiple columns if we simply export the DF

for i in cln_txt_DF.index:
  with open(f'{default_directory}/datafiles/text_cln/art_cln_{i}.txt', 'w', errors='ignore') as file:
    file.write(str(cln_txt_DF.loc[i,'text']))
    file.close()

#Then import the text back in as individual text files
final_cln_text_info = []
for i in range(0,len(cln_txt_DF)):
  with open(f'{default_directory}/datafiles/text_cln/art_cln_{i}.txt') as file:
    final_cln_text_info.append(file.read().strip('\n'))
    file.close()

#Now create a new dataframe that uses the cleaned text. Also removed any remaining rows have no article text.
art_price_cln_DF=art_price_DF.copy(deep=True)
art_price_cln_DF['cln_txt']=final_cln_text_info
art_price_cln_DF=art_price_cln_DF[art_price_cln_DF['cln_txt']!='']

art_price_cln_DF

"""#Now go back to running models with the cleaned text data."""

# model 5

max_len=300
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(64))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

# decrease max_len, lstm
# model 6
max_len=150
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increaes max_len
# model 7
max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.9,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increase LSTM
# model 8

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(256))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.9,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#Reduce output dim, lstm, no max len
# model 9

tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = 500,output_dim = 8,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(8))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#Add back max_len, increase input output lstm
# model 10

max_len=150
num_words=5000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 32,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#Increaes max_len and input
# model 11

max_len=150
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 32,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#Increase output
# model 12

max_len=150
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increase max_len
# model 13

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increase lstm
# model 14

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(256))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#decrease test_size
# model 15

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(256))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#try changing activation to softmax from sigmoid
# model 16

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'softmax')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#try changing activation to tanh from sigmoid
# model 17

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'tanh')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#changed loss to categorical_crossentropy
# model 18

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#loss back to bin, epochs up to 100
# model 19

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(256))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#reduce lstm
# model 20

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a bi-directional LSTM layer
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.5,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

"""#Now try adding in a CNN to see if I can get the accuracies up."""

# model 21

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 32,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increase to 500 epochs
# model 22

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 32,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 500,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#epochs to 50, increase max input output
# model 23

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 50,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#reduce output dim, lstm, max, input
# model 24

max_len=150
num_words=5000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 32,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(32))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#add a second dense layer
# model 25

max_len=150
num_words=5000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 32,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(32))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'relu')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increase lstm
# model 26

max_len=150
num_words=5000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 32,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'relu')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#change the activation function in the 64 neuron dense layer
# model 27

max_len=150
num_words=5000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 32,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increase output dims to 256
# model 28

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#try with 500 epochs
# model 29

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 500,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

"""#Now try pretrained word embeddings instead of "self"
"""

###First download the glove embedder. This took a while to run so the file will be provided in the project.
#!python -m wget http://nlp.stanford.edu/data/glove.6B.zip

#extract the file
patoolib.extract_archive(f'{default_directory}/embedders/glove.6B.zip')

#explore the file
with open(f'{default_directory}/embedders/glove.6b/glove.6B.100d.txt', 'r',encoding='UTF8') as f:
  lines = f.readlines()
  for i in [0, 9, 99, 999, 99999, 399999]:
    print(lines[i].strip())

#create the embedding index using the glove embedder
path_to_glove_file = f'{default_directory}/embedders/glove.6b/glove.6B.200d.txt'

embeddings_index = {}
with open(path_to_glove_file,encoding='UTF8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print(f'Found {len(embeddings_index)} word vectors.')

# Create an embedding matrix
max_len=150
num_words=10000
embedding_dim=200

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < num_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

#create the embedding layer
embedding_layer = layers.Embedding(
    num_words,
    embedding_dim,
    embeddings_initializer = keras.initializers.Constant(embedding_matrix),
    trainable = False,
    mask_zero = True,
)

#now create the model
# model 30

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 50,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increase the number of epochs
# model 31

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increase the max_len and num_words
max_len=200
num_words=20000
embedding_dim=200

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

# Create an embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < num_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

embedding_layer = layers.Embedding(
    num_words,
    embedding_dim,
    embeddings_initializer = keras.initializers.Constant(embedding_matrix),
    trainable = False,
    mask_zero = True,
)

#now train the model
# model 32
X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 50,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#try it with 1 epoch (this is not included in final results)
# model 33
X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 1,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

"""#Now try fasttext embedder"""

#This was taking a while and then may have frozen on me, so I manually went to the link it
# listed that it was downloading from and downloaded the file. I will include it
# with the project files.

#fasttext.util.download_model('en', if_exists='ignore')

#exctract the fasttext embedder
with gzip.open(f'{default_directory}/embedders/cc.en.300.bin.gz', 'rb') as f_in:
    with open(f'{default_directory}/embedders/cc.en.300.bin', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

#set the embedder
fasttext_model = FastText.load_fasttext_format(f'{default_directory}/embedders/cc.en.300.bin')

#create the embedding matrix and embedding layer
def get_word_embedding(word):
    try:
        return fasttext_model.wv.get_vector(word)
    except KeyError:
        # If the word is not in the vocabulary, return a vector of zeros
        return np.zeros(fasttext_model.vector_size)

# Vocabulary size and embedding dimension
vocab_size = len(fasttext_model.wv.index_to_key)
embedding_dim = fasttext_model.vector_size

# Create an embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for i, word in enumerate(fasttext_model.wv.index_to_key):
    embedding_matrix[i] = get_word_embedding(word)

# Create an Embedding layer in Keras
embedding_layer = Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            trainable=False)

#now build a model with the fasttext embedding layer
# model 34

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (1,), dtype ='int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
#x=layers.Conv1D(128, 5, activation='relu')(embedded)
#x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(embedded)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 10,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increaes epochs
# model 35

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (1,), dtype ='int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
#x=layers.Conv1D(128, 5, activation='relu')(embedded)
#x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(embedded)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#####Try adding the masking back in
def get_word_embedding(word):
    try:
        return fasttext_model.wv.get_vector(word)
    except KeyError:
        # If the word is not in the vocabulary, return a vector of zeros
        return np.zeros(fasttext_model.vector_size)

# Vocabulary size and embedding dimension
vocab_size = len(fasttext_model.wv.index_to_key)
embedding_dim = fasttext_model.vector_size

# Create an embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for i, word in enumerate(fasttext_model.wv.index_to_key):
    embedding_matrix[i] = get_word_embedding(word)

# Create an Embedding layer in Keras
embedding_layer = Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            trainable=False,
                            mask_zero = True)

#now the model
# model 36

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (1,), dtype ='int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
#x=layers.Conv1D(128, 5, activation='relu')(embedded)
#x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(embedded)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 50,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#change the comp optimizer back to rmsprop since we aren't using CNN here
# model 37

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (1,), dtype ='int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
#x=layers.Conv1D(128, 5, activation='relu')(embedded)
#x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(embedded)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 50,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increase epochs
# model 38

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (1,), dtype ='int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
#x=layers.Conv1D(128, 5, activation='relu')(embedded)
#x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(embedded)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#see if changing the shape allows CNN to work with fast
# model 39

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype ='int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(embedded)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#remove CNNs again
# model 40

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (1,), dtype ='int64')

embedded = embedding_layer(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
#x=layers.Conv1D(128, 5, activation='relu')(embedded)
#x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(embedded)
x = layers.Dropout(0.5)(x)
#x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#go back to a self embedding layer
# model 41
max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#take the model that has had the best accuracy so far and adjust the learning rate to .01
# model 42

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.01
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#learning rate of .0005, testing split of .5
# model 43

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.5)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0005
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#learning rate back to .0001, now try to add some L2 regularization of .01
# model 44

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu',kernel_regularizer=regularizers.l2(0.01))(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh',kernel_regularizer=regularizers.l2(0.01))(x)
outputs = layers.Dense(1, activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.01))(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#L2 regularization of .0001
# model 45

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh',kernel_regularizer=regularizers.l2(0.0001))(x)
outputs = layers.Dense(1, activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.0001))(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#take the model that has had the best accuracy so far and adjust the learning rate to .0001
# model 46

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#drop the learning rate down to .00001, drop back down max_len and num_words
# model 47

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.00001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#increase the learning rate to .0005
# model 48

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0005
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#drop the learning rate down to .00005
# model 49

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.00005
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#learning rate of .00009
# model 50

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.00009
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#learning rate back to .0001, now try to add some L2 regularization of .001
# model 51

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu',kernel_regularizer=regularizers.l2(0.001))(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh',kernel_regularizer=regularizers.l2(0.001))(x)
outputs = layers.Dense(1, activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.001))(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#L2 regularization of .1
# model 52

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu',kernel_regularizer=regularizers.l2(0.1))(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh',kernel_regularizer=regularizers.l2(0.1))(x)
outputs = layers.Dense(1, activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.1))(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#learning rate of .00009, test set of .2
# model 53

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.00009
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#learning rate of .00008
# model 54

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.00008
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#learning rate of .0001, raise max_len to 200
# model 55

max_len=200
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#learning rate of .0001, num_words to 20000
# model 56

max_len=200
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#learning rate of .0001, drop max_len to 150
# model 57

max_len=150
num_words=20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#run the best model with uncleaned data, now 500 epochs
# model 58

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_DF['art_text'])
sequences = tokenizer.texts_to_sequences(art_price_DF['art_text'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 500,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')
#top acc: .9601
#top val: .6476
#top acc: .627

#see how adjusting the testing size to .5 affects the model
# model 59

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.5)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#see how adjusting the testing size to .2 with a learning rate of .0001 affects the model
# model 60

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#use regularization of .001 but don't use it on the output layer
# model 61

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu',kernel_regularizer=regularizers.l2(0.001))(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh',kernel_regularizer=regularizers.l2(0.001))(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#run regularization on just one layer
# model 62

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.3)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh',kernel_regularizer=regularizers.l2(0.001))(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#run the best model with uncleaned data and .0001 learning rate
# model 63

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_DF['art_text'])
sequences = tokenizer.texts_to_sequences(art_price_DF['art_text'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#run best model uncleaned data with regularization
# model 64

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_DF['art_text'])
sequences = tokenizer.texts_to_sequences(art_price_DF['art_text'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu',kernel_regularizer=regularizers.l2(0.001))(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh',kernel_regularizer=regularizers.l2(0.001))(x)
outputs = layers.Dense(1, activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.001))(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 100,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#see how adjusting the testing size to .2 with 500 epochs affects the model
# model 65

max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 500,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

#test size at .2, .0001 learning rate, 500 epochs
# model 66
max_len=150
num_words=10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(art_price_cln_DF['cln_txt'])
sequences = tokenizer.texts_to_sequences(art_price_cln_DF['cln_txt'])
tok_art_text = pad_sequences(sequences,maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tok_art_text, art_price_cln_DF['inc_factor'], test_size=0.2)

inputs = keras.Input(shape = (None,), dtype = 'int64')

embedded = layers.Embedding(input_dim = num_words,output_dim = 256,mask_zero = True)(inputs)

# the embeddings we learned become the inputs
# to a CNNLSTM layer
x=layers.Conv1D(128, 5, activation='relu')(embedded)
x=layers.MaxPooling1D(5)(x)

x = layers.Bidirectional(layers.LSTM(100))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation = 'tanh')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

initial_learning_rate = 0.0001
optimizer1 = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer = optimizer1,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('embeddings_bidir_gru.keras')#,
                                    #save_best_only = True)<-----these options no longer work
]

model.fit(x=X_train,
          y=y_train,
          validation_split = 0.2,
          epochs = 500,
          callbacks = callbacks)

model = keras.models.load_model('embeddings_bidir_gru.keras')
print(f'Test acc: {model.evaluate(x=X_test,y=y_test)[1]:.3f}')

"""################################################################################
### Now perform the analysis ###################################################
################################################################################
"""

#All of the accuracies and different variations from all of tests were manually collected and compiled into one excel sheet for analysis
# It will now be imported

acc_review=pd.read_excel(f'{default_directory}/datafiles/accuracy review.xlsx')
acc_review

#some of the columns should be factors, converting them now
acc_review['cln_fac']=pd.factorize(acc_review['cleaned'])[0]
acc_review['mask_fac']=pd.factorize(acc_review['Mask'])[0]
acc_review['emb_fac']=pd.factorize(acc_review['Embedding'])[0]
acc_review['CNN_fac']=pd.factorize(acc_review['CNN'])[0]
acc_review['den_fac']=pd.factorize(acc_review['dense_act'])[0]
acc_review['2den_fac']=pd.factorize(acc_review['2nd_dense_act'])[0]
acc_review['opt_fac']=pd.factorize(acc_review['comp_opt'])[0]
acc_review['loss_fac']=pd.factorize(acc_review['comp_loss'])[0]

#drop the original columns that were factorized, then rearrange the columns
acc_rev_fac=acc_review.copy(deep=True)
acc_rev_fac=acc_rev_fac.drop(['cleaned','Mask','Embedding','CNN','dense_act','2nd_dense_act','comp_opt','comp_loss'],axis=1)
cols_move=acc_rev_fac.columns.tolist()
len(cols_move)
cols_move=cols_move[0:10]+cols_move[13:21]+cols_move[10:13]
len(cols_move)
acc_rev_fac=acc_rev_fac[cols_move]

#now look at correlations of the different variations in the models to the accuracies, then creat a heatmap out of it
acc_rev_fac_cor=acc_rev_fac.corr()
sns.set(rc={'figure.figsize':(20,15)},font_scale=1.4)
heat_acc_rev_fac=sns.heatmap(acc_rev_fac_cor,annot=True,cmap='BrBG',fmt='.2f',annot_kws={'size': 15})
print(heat_acc_rev_fac)
plt.savefig(f'{default_directory}/datafiles/images/heat_acc_rev_fac.jpg')

#now look at how adjusting the learning rate versus the number of epochs affected the accuracy
no_learn_epoch = []
with open(f'{default_directory}/datafiles/no_learn_epoch.txt',encoding="UTF-8") as file:
    for line in file:
        no_learn_epoch.append(line.strip('\n'))
no_learn_epoch=[float(l) for l in no_learn_epoch if l != '']

learn_epoch = []
with open(f'{default_directory}/datafiles/learn_epoch.txt',encoding="UTF-8") as file:
    for line in file:
        learn_epoch.append(line.strip('\n'))
learn_epoch=[float(l) for l in learn_epoch if l != '']

len(no_learn_epoch)
len(learn_epoch)

learn_epoch_df=pd.DataFrame()
learn_epoch_df['epochs']=float(learn_epoch)

no_learn_epoch_df=pd.DataFrame()
no_learn_epoch_df['epochs']=no_learn_epoch

fig, ax = plt.subplots(figsize=(20,15))
plt.plot(range(1,101,1),no_learn_epoch,color='red',label='.001 Learn Rate')
plt.plot(range(1,101,1),learn_epoch,color='blue',label='.0001 Learn Rate')
plt.xlabel("Epoch",fontsize=24)
plt.ylabel("Accuracy",fontsize=24)
plt.title('Learning Rate Accuracy', pad=20, fontsize=30)
plt.xticks(range(0,101,10),fontsize=20)
plt.yticks(fontsize=20)
fig.tight_layout()
plt.legend(fontsize=20)

#now look at how the cnn layers affected the accuracies
no_cnn_epoch = []
with open(f'{default_directory}/datafiles/no_cnn_epoch.txt',encoding="UTF-8") as file:
    for line in file:
        no_cnn_epoch.append(line.strip('\n'))
no_cnn_epoch=[float(l) for l in no_cnn_epoch if l != '']

cnn_epoch = []
with open(f'{default_directory}/datafiles/cnn_epoch.txt',encoding="UTF-8") as file:
    for line in file:
        cnn_epoch.append(line.strip('\n'))
cnn_epoch=[float(l) for l in cnn_epoch if l != '']

len(no_cnn_epoch)
len(cnn_epoch)

cnn_epoch_df=pd.DataFrame()
cnn_epoch_df['epochs']=cnn_epoch

no_cnn_epoch_df=pd.DataFrame()
no_cnn_epoch_df['epochs']=no_cnn_epoch

fig, ax = plt.subplots(figsize=(20,15))
plt.plot(range(1,101,1),no_cnn_epoch,color='red',label='No CNN')
plt.plot(range(1,101,1),cnn_epoch,color='blue',label='CNN')
plt.xlabel("Epoch",fontsize=24)
plt.ylabel("Accuracy",fontsize=24)
plt.title('CNN Effect On Accuracy', pad=20, fontsize=30)
plt.xticks(range(0,101,10),fontsize=20)
plt.yticks(fontsize=20)
fig.tight_layout()
plt.legend(fontsize=20)