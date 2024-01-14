"""Here we have functions to prepare, refresh or clean our data"""
import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pymysql
from geopy.geocoders import Nominatim
import tqdm

#Load API keys
load_dotenv()
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')
SQL_PWD =  os.getenv('SQL_PWD')
SQL_USER = os.getenv('SQL_USER')
SQL_DB = os.getenv('SQL_DB')
SQL_HOST = os.getenv('SQL_HOST')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#All functions related to translating text.
#Deepl API

#Translate a single instance from german to english
def translate2EN(text):
  api_url = "http://api-free.deepl.com/v2/translate"
  response = requests.get(api_url,
                        params={
    "auth_key": DEEPL_API_KEY,
    "source_lang": "DE",
    "target_lang": "EN",
    "text": text,
  }) # type: ignore
  return response.json()["translations"][0]["text"]

#Translate a single instance from english to german
def translate2DE(text):
  api_url = "http://api-free.deepl.com/v2/translate"
  response = requests.get(api_url,
                        params={
    "auth_key": DEEPL_API_KEY,
    "source_lang": "EN",
    "target_lang": "DE",
    "text": text,
  }) # type: ignore
  return response.json()["translations"][0]["text"]

# Will use the Deepl API to translate the colum f_description to english.
# Optionally can save a csv at savefile location
def translateCol2EN (data, backup=False, savefile=""):
  data_out=data.copy()
  if 'description_en' not in data.columns:
    data_out["description_en"] = ""
  #detect missing translations
  for index in np.where(pd.Series(data_out['description_en'].isnull()) | 
                pd.Series(data_out['description_en'] == "nan"))[0]:
    #print(index)
    try:
      print("Translating row " + str(index))
      data_out.loc[index,'description_en'] = translate2EN(data_out.loc[index,'f_description'])
    except ValueError as e:
      print("Error in row " + str(index))
  data_out["description_en"]=data_out["description_en"].apply(str)
  if backup and savefile:
    data_out.to_csv(savefile, sep=",", index=False)
  return data_out

#Function(s) to update the data.
#Used to refresh the data for Task1
# 1: Query to server
# 2: Delete rows without description
# 3: Transfer existing translations
# 4: Create new and missing translations
# 5: Save new result
def refreshTask1(savepath):
  with open('Prompts/Query1.sql', 'r') as sqlfile:
    task1 = sqlfile.read()

  connection = pymysql.connect(host=SQL_HOST, port=3306, user=SQL_USER, passwd=SQL_PWD, database=SQL_DB)
  
  with connection:
    with connection.cursor() as cursor:
      cursor.execute(task1)
      rows = cursor.fetchall()

  # Convert the list of tuples to a Pandas DataFrame
  data = pd.DataFrame(rows, columns=[col[0] for col in cursor.description])
  
  #REMIND: Delete rows without description and duplicates
  data = data[~data.f_description.isnull()]
  data = data.drop_duplicates(subset='f_key', keep='first', ignore_index=True)

  
  #Add existing english descriptions
  newData = transferTranslation(data, savepath)

  #Create new and missing translations
  newData = translateCol2EN(newData, backup=False)

  #Add regional info
  SetRegion(newData)

  #save new result
  newData.to_csv(savepath + "Datasets/ConnactzTask1.csv", sep=",", index=False)
  print("Data refreshed, descriptions updated and saved")
  return newData

#English descriptions are not stored in the database.
#If we have an old file we will use that to add existing translated descriptions to new files
#The band name and german description need to be the same
def transferTranslation(data, savepath):
  #create backup
  os.replace(savepath + "Datasets/ConnactzTask1.csv", savepath + "Datasets/ConnactzTask1_Old.csv")
  #probably unnecessary to reload the data but anyways...
  data_old =  pd.read_csv(savepath + "Datasets/ConnactzTask1_Old.csv", sep=",", index_col=False)
  newDF = pd.merge(data,data_old[["f_name","f_description","description_en"]],on=["f_name","f_description"], how="left")
  newDF.to_csv(savepath + "Datasets/ConnactzTask1.csv", sep=",", index=False)
  print("Transferred Translations successfully")
  return newDF


#TODO: remove <p><strong> and other control characters
def createTriplets(data, backup=False, filepath=""):
  #create new dataframe with all triples in different columns then assemble them with &&
  name = pd.Series('band | name | ' + data.f_name.str.replace(' ','_'), name='name')
  loc = pd.Series(data.f_name.str.replace(' ','_') +' | locality | '+ data.f_locality.str.replace(' ','_'), name='loc')
  dist = pd.Series(data.f_name.str.replace(' ','_') +' | distance | '+ data.f_distance.astype("string"), name='dist')
  cast = pd.Series(data.f_name.str.replace(' ','_') +' | cast | '+ data.f_casts.str.replace(',','_'), name='cast')
  genre = pd.Series(data.f_name.str.replace(' ','_') +' | genre | '+ data.f_genres.str.replace(',','_'), name='genre')
  eventtype = pd.Series(data.f_name.str.replace(' ','_') +' | eventtype | '+ data.f_eventtypes.str.replace(',','_'), name='eventtype')
  data_triples = pd.concat([name,loc,dist,cast,genre,eventtype], axis=1)
  #since some fields are empty we might have a slight problem in the next step with "&& &&" type of strings.
  data_triples['input_text'] = data_triples[['name', 'loc', 'dist', 'cast', 'genre', 'eventtype']].fillna('').agg(' && '.join, axis=1)

  data_triples['target_text_de'] = data.f_description
  data_triples['target_text_en'] = data.description_en

  if backup and filepath:
    #Save
    data_triples.to_csv(filepath, sep=",", index=False) # type: ignore

  return data_triples

def detectLanguage(data):
  #TODO: Implement detectLanguage
  return

def SetRegion(data):
  """Uses Nominatim and the coordinates to set the region of the band.
  No return value, the region is added directly to the dataframe.
  """ 

  def geolocate(row):
    try:
      address = geolocator.reverse((row['f_locality_lat'], row['f_locality_lon'])).address
      return address.split(', ')[-3]  # The region is the third last entry in the address
    except:
      return None
  
  # Create a geolocator object
  geolocator = Nominatim(user_agent='alex.mercier@tum.de')

  # Reverse geocode each coordinate and add the address to the DataFrame
  print("Adding regions")
  tqdm.tqdm.pandas()
  data['region'] = data.progress_apply(geolocate, axis=1)
  return