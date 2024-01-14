# G-Eval

import json
import openai
import time
import tqdm
import nltk
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import re
import nltk
from nltk.stem import SnowballStemmer
from termcolor import colored


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


#TODO: implement token usage stats
#Previously used gpt-3.5-turbo-0301
def run_gpt4_eval(ResultsDF, Evalprompt, EvalCol, model='gpt-4-1106-preview', max_score=5, replaceSecond=None, invalidVal = 0):
  """Runs the GPT-4 model on the given data and returns the result as a json.
  ResultsDF: The dataframe containing the data to be evaluated
  Evalprompt: The prompt that will be loaded for evaluation
  EvalCol: The column of ResultDF that will hold the results
  model: The model to use for evaluation
  max_score: The maximum score that can be given by the model (Used for filtering out bad responses)
  replaceSecond: (optional) A second replacement can be specified. It will take it's replacement from ResultsDF.Parameters.[replaceSecond] (should be a Series of Dict)
  invalidVal: (optional) The value that will be used for invalid responses

  Bad responses are logged as soon as one value is out of bounds. If at least 10 responses are valid, the score is still kept.

  Returns: A DF containing the rows that failed to be evaluated
  """
  openai.api_key = OPENAI_API_KEY
  #summeval = EvalCol
  prompt = open(Evalprompt).read()
  
  

  #log bad responses
  bad_responses = pd.DataFrame({'f_key': pd.Series(dtype='str'),
                                'Experiment': pd.Series(dtype='str'),
                                'Answer': pd.Series(dtype='str'),
                                'Response': pd.Series(dtype='str'),
                                'ResClean': pd.Series(dtype='str'), #cleaned responses
                                'Score':pd.Series(dtype='float'),
                                'Error':pd.Series(dtype='bool')
                              })


  for instance in tqdm.tqdm(np.where(pd.Series(ResultsDF[EvalCol].isnull()))[0]):
    #source = instance['source']
    #system_output = instance['system_output']
    cur_prompt = prompt.replace('{{Description}}', ResultsDF.loc[instance, 'Answer']) #.replace('{{Document}}', source)
    if replaceSecond:
      cur_prompt = cur_prompt.replace('{{SecondReplacement}}', ResultsDF.loc[instance, 'Parameters'][0][replaceSecond])
    #instance['prompt'] = cur_prompt

    #wait time when openai is lazy/I reach the limits
    wait = 2

    while True:
      try:
        _response = openai.ChatCompletion.create(
          model=model,
          messages=[{"role": "system", "content": cur_prompt}],
          temperature=1.3,
          max_tokens=2,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None,
          # logprobs=40,
          n=20
        )
        #time.sleep(0.5)
        all_responses = [_response['choices'][i]['message']['content'] for i in range(len(_response['choices']))]
        #filter non-int answers and converts to int
        #filter untrustworthy answers
        all_responses2= [ int(x) for x in all_responses if x.isdigit() and int(x) >= 0 and int(x) <= max_score ]
        #average over the responses
        res_avg = sum(all_responses2)/len(all_responses2)

        #log bad responses
        if res_avg < 0 or res_avg > max_score or len(all_responses2) < 15:
          #print("Bad response: ", instance)
          bad_responses = pd.concat([bad_responses, 
                                      pd.DataFrame({'f_key': [ResultsDF.loc[instance, 'f_key']],
                                                    'Experiment': [ResultsDF.loc[instance, 'Experiment']],
                                                    'Answer': [ResultsDF.loc[instance, 'Answer']],
                                                    'Response': ','.join(str(v) for v in all_responses),
                                                    'ResClean': ','.join(str(v) for v in all_responses2),
                                                    'Score': [res_avg],
                                                    'Error':[False]})
                                    ], ignore_index=True)
          #Set invalid value
          ResultsDF.loc[instance, EvalCol] = invalidVal
        else:        
          #To get an accurate average we want at least 15 valid responses
          ResultsDF.loc[instance, EvalCol] = res_avg
        #XXX: GPT4 has speed limits, so we need to wait a bit
        time.sleep(0.1)
        break
      except Exception as e:
        if ("limit" in str(e)) or ("overloaded" in str(e)):
          print("waiting " +wait+" seconds")
          time.sleep(wait)
          wait = wait * 2
        else:
          # Handle other exceptions here
          print("Row: ", instance)
          print("Error:", e)
          bad_responses = pd.concat([bad_responses, 
                                      pd.DataFrame({'f_key': [ResultsDF.loc[instance, 'f_key']], 
                                                    'Experiment': [ResultsDF.loc[instance, 'Experiment']],
                                                    'Answer': [ResultsDF.loc[instance, 'Answer']], 
                                                    'Response': ','.join(str(v) for v in all_responses), 
                                                    'ResClean': ','.join(str(v) for v in all_responses2),
                                                    'Score': [invalidVal], 
                                                    'Error':[True]})
                                    ], ignore_index=True)
          #Set invalid value
          ResultsDF.loc[instance, EvalCol] = invalidVal
          break
  return bad_responses

def TTR(text):
  """A very simple/simplistic mearsure of lexical density: Type-Token-Ratio"""
  tokens = nltk.word_tokenize(text.lower())
  types = set(tokens)
  ttr = len(types) / len(tokens)
  return ttr

def eval(ResultsDF, savepath, model='gpt-4-1106-preview'):
  print("Running G-Eval Engagingness")
  br_en = run_gpt4_eval(ResultsDF, savepath + '/Prompts/g_eval_eng.txt', 'Geval_eng', max_score=5, model=model)
  print("Running G-Eval Fluidity")
  br_flu =  run_gpt4_eval(ResultsDF, savepath + '/Prompts/g_eval_flu.txt', 'Geval_flu', max_score=3, model=model)
  print("Running G-Eval Naturalness")
  br_nat = run_gpt4_eval(ResultsDF, savepath + '/Prompts/g_eval_nat.txt', 'Geval_nat', max_score=3, model=model)
  print("Running G-Eval Informativeness")
  br_inf = run_gpt4_eval(ResultsDF, savepath + '/Prompts/g_eval_inf.txt', 'Geval_inf', max_score=4, replaceSecond='gendata', model=model)
  print("Running G-eval Quality ")
  br_qua = run_gpt4_eval(ResultsDF, savepath + '/Prompts/g_eval_qua.txt', 'Geval_qua', max_score=5, model=model)

  ResultsDF['TTR'] = ResultsDF['Answer'].apply(TTR)

  #Prepare and return the bad responses
  #note: suffixes are only applied to duplicate columns
  merged_br = pd.merge(br_en, br_flu, on=['f_key','Experiment','Answer'], how='outer', suffixes=('_en', ''))
  merged_br = pd.merge(merged_br, br_nat, on=['f_key','Experiment','Answer'], how='outer', suffixes=('_flu', ''))
  merged_br = pd.merge(merged_br, br_inf, on=['f_key','Experiment','Answer'], how='outer', suffixes=('_nat', ''))
  merged_br = pd.merge(merged_br, br_qua, on=['f_key','Experiment','Answer'], how='outer', suffixes=('_inf', '_qua'))
  overall_score(ResultsDF)
  return merged_br

#REMIND: Code to rerun the eval for -1 values
# cols =['Geval_eng', 'Geval_flu', 'Geval_nat', 'Geval_inf', 'TTR',
#        'OverallScore']
# ResultsDF[cols] = ResultsDF[cols].replace(-1, np.nan)

gendata_base = """name = {{f_name}},
hometown = {{f_locality}},
travel distance = {{f_distance}}km,
genres = {{f_genres}},
event types = {{f_eventtypes}}"""

def addGendata2ResultsDF(ResultsDF, task1_in):
  """Add generation data to ResultsDF.Parameters column. This is a workaround to avoid having to re-run the whole evaluation process.
  This doesn't overwrite or delete other values stored in the Parameters column."""

  for index, row in ResultsDF.iterrows():
    bandindex = task1_in.loc[task1_in.f_key == row.f_key].index[0]
    gendata = gendata_base.replace('{{f_name}}', str(task1_in.loc[bandindex, 'f_name'])).replace('{{f_locality}}', str(task1_in.loc[bandindex, 'f_locality'])).replace('{{f_distance}}', \
    str(task1_in.loc[bandindex, 'f_distance'])).replace('{{f_genres}}', task1_in.loc[bandindex, 'f_genres']).replace('{{f_eventtypes}}', task1_in.loc[bandindex, 'f_eventtypes'])
    
    #TODO: Verify that empty cell is handled
    try:
      parameters = ResultsDF.loc[index, 'Parameters'][0]
    except:
      parameters = {}
    parameters['gendata'] = gendata
    ResultsDF.loc[index, 'Parameters'] = [parameters]


def overall_score(ResultsDF, min=0):
  """(Re-)Calculates the overall score for each row in ResultsDF.
  Normalizes the scores to a range of 0 to 1.
  ResultsDF: The dataframe containing the data to be evaluated
  """
  #-1 everywhere as 0 is not included in the scores. Maybe a dumb decision now that I think about it
  ResultsDF['OverallScore'] = ((ResultsDF['Geval_eng']-min)/(5-min) + (ResultsDF['Geval_flu']-min)/(3-min) + (ResultsDF['Geval_nat']-min)/(3-min) + (ResultsDF['Geval_inf']-min)/(4-min)) / 4
##


from nltk import ngrams
from collections import Counter
def common_ngrams(ResultsDF, Experiment, n={2,3,4}):
  """Count all occurring n-grams, for all texts for an experiment.
  returns the top 100
  """
  stemmer = SnowballStemmer('german')

  jaccPrep = ResultsDF.loc[ResultsDF['Experiment'] == Experiment, ['Answer']].copy()
  #lowercase, remove punctuation, tokenize, stem
  jaccPrep['Stemm'] = jaccPrep['Answer'].apply(lambda x: [stemmer.stem(word) for word in nltk.word_tokenize(re.sub(r'[^a-zA-Z0-9 ]','',x.lower()))])

  #create all ngrams
  jaccPrep['n_gram'] = jaccPrep['Stemm'].apply(lambda x: [ngram for n_value in n for ngram in ngrams(x, n_value)])

  #count the occurrences of each ngram
  ngram_counts = Counter(ngram for ngrams_list in jaccPrep['n_gram'] for ngram in ngrams_list)

  #return the top 100 most common ngrams
  return ngram_counts.most_common(100)

  

#NOTE: does average make sense?
#TODO: Add cosine similarity
def jaccard_sim_score(ResultsDF, Experiment, n={2,3,4}):
  """Calculate the jaccard similarity for all texts for an experiment.
  Result will be a matrix as well as the average.
  Currently only SnowballStemmer english is supported.
  """
  stemmer = SnowballStemmer('german')

  jaccPrep = ResultsDF.loc[ResultsDF['Experiment'] == Experiment, ['Answer']].copy()
  #lowercase, remove punctuation, tokenize, stem
  jaccPrep['Stemm'] = jaccPrep['Answer'].apply(lambda x: [stemmer.stem(word) for word in nltk.word_tokenize(re.sub(r'[^a-zA-Z0-9 ]','',x.lower()))])

  #create all ngrams
  jaccPrep['n_gram'] = jaccPrep.apply(lambda x: set(), axis=1)
  for k in n:
    for row in jaccPrep.itertuples():
      row.n_gram.update( set([' '.join(row.Stemm[i:i+k]) for i in range(len(row.Stemm)-k+1)]) )
  
  def jaccard_similarity(s1_ngrams, s2_ngrams):
    try:
      return len(s1_ngrams.intersection(s2_ngrams)) / len(s1_ngrams.union(s2_ngrams))
    except ZeroDivisionError:
      print("ZeroDivisionError: setting to 0")
      return 0

  jaccard_matrix = pd.DataFrame(index=jaccPrep.index, columns=jaccPrep.index)
  for i in range(len(jaccPrep.index)):
    for j in range(i+1, len(jaccPrep.index)):
        jaccard_matrix.iloc[i, j] = jaccard_similarity(jaccPrep.iloc[i]['n_gram'], jaccPrep.iloc[j]['n_gram'])
        jaccard_matrix.iloc[j, i] = jaccard_matrix.iloc[i, j]

  # Replace diagonal with NaN
  np.fill_diagonal(jaccard_matrix.values, np.nan)

  # Calculate mean excluding NaN values
  mean_value = jaccard_matrix.mean().mean()

  return jaccard_matrix, mean_value

def inspectSimMatrix(matrix, ResultsDF, top=10):
  """Inspect the most similar texts in a matrix.
  Converts matrix to a stacked DataFrame and returns the top n values.
  """
  # Stack the DataFrame and reset the index
  stacked = matrix.stack().reset_index()

  # Rename the columns
  stacked.columns = ['Row', 'Column', 'Value']

  # Convert 'Value' column to numeric
  stacked['Value'] = pd.to_numeric(stacked['Value'], errors='coerce')

  #Only keep the lower triangle of the original matrix
  stacked = stacked[stacked['Row'] < stacked['Column']]

  # Get the top 10 highest values
  top_10 = stacked.nlargest(top, 'Value')

  # Get the indexes as tuples
  indexes = list(zip(top_10['Row'], top_10['Column'], top_10['Value']))

  inspector = pd.DataFrame([(ResultsDF.at[ix, 'f_key'], ResultsDF.at[ix, 'Answer'], ResultsDF.at[iy, 'f_key'], ResultsDF.at[iy, 'Answer'], score) for (ix, iy, score) in indexes],
                      columns=['f_key1', 'Answer1', 'f_key2', 'Answer2', 'Score'])
  return inspector

def highlightNgram(text1, text2, max_n=4):
  """Compares 2 texts and highlights the ngrams that are the same in a print output.
  The ngram comparison is done on the stemmed words.
  The output uses the whole sentence. Special characters can lead to errors if the stemmed version has a differing number of words.
  """
  stemmer = SnowballStemmer('german')

  def generate_ngrams(s, n):
    # Convert the string to a list of words
    words = [stemmer.stem(word) for word in nltk.word_tokenize(re.sub(r'[^a-zA-Zäöü0-9 ]','',re.sub(r'\n',' ',s.lower())))]
    #words = s.split(' ')
    # Generate the n-grams
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    return ngrams

  # stemmatized ngrams for each input
  ngrams_s1 = {n: generate_ngrams(text1, n) for n in range(2, max_n+1)}
  ngrams_s2 = {n: generate_ngrams(text2, n) for n in range(2, max_n+1)}
  # print(ngrams_s1[2])

  # Print the strings with colored n-grams
  for n in range(2,max_n+1):
    print('n =', n)

    # Split the original strings into words (with minimal modifications)
    s1_wo = (re.sub(r' [^a-zA-Zäöü0-9 ] ',' ',re.sub(r'\n',' ',text1))).split(' ')
    s1_wo = [word for word in s1_wo if word != '']
    s2_wo = (re.sub(r' [^a-zA-Zäöü0-9 ] ',' ',re.sub(r'\n',' ',text2))).split(' ')
    s2_wo = [word for word in s2_wo if word != '']
    # Coloring arrays
    col_s1 = [False] * len(s1_wo)
    col_s2 = [False] * len(s2_wo)

    # Check if both arrays have the same length
    if len(ngrams_s1[n]) != len(s1_wo) - n + 1:
      print('Error: ngrams_s1['+str(n)+'] has length', len(ngrams_s1[n]), 'but should have length', len(s1_wo) - n + 1)
      return
    if len(ngrams_s2[n]) != len(s2_wo) - n + 1:
      print('Error: ngrams_s2['+str(n)+'] has length', len(ngrams_s2[n]), 'but should have length', len(s2_wo) - n + 1)
      return

    # Check if the n-grams are in the other string
    for w in range(len(s1_wo)-n+1):
      if ngrams_s1[n][w] in ngrams_s2[n]:
        col_s1[w:w+n] = [True] * n
    # Print the strings with colored n-grams
    for w in range(len(s1_wo)):
      if col_s1[w]:
        print(colored(s1_wo[w], 'green'), end=' ')
      else:
        print(s1_wo[w], end=' ') 
    print("\n")

    # Repeat for the second string
    for w in range(len(s2_wo)-n+1):
      if ngrams_s2[n][w] in ngrams_s1[n]:
        col_s2[w:w+n] = [True] * n
    for w in range(len(s2_wo)):
      if col_s2[w]:
        print(colored(s2_wo[w], 'red'), end=' ')
      else:
        print(s2_wo[w], end=' ') 
    print("\n\n")

############# NOT IN USE #############
# I had hope to accelerate the code by vectorizing the evaluation, but it seems that the API is not able to handle it
# I will leave the code here in case it is useful in the future
# The bad_responses are not properly logged in this version

# def evaluate_row(row, prompt, model='gpt-3.5-turbo-0301', max_score=5):
#   """Evaluates a single row using the GPT-4 model and returns the result as a string.
#   row: The row to be evaluated
#   prompt: The prompt that will be loaded for evaluation
#   model: The model to use for evaluation
#   max_score: The maximum score that can be given by the model (Used for filtering out bad responses)

#   Returns: A string containing the evaluation result
#   """
#   cur_prompt = prompt.replace('{{Description}}', row['Answer']) #.replace('{{Document}}', source)

#   # Evaluate the row using the GPT-4 model
#   _response = openai.ChatCompletion.create(
#     model=model,
#     messages=[{"role": "system", "content": cur_prompt}],
#     temperature=2,
#     max_tokens=1,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop=None,
#     # logprobs=40,
#     n=20
#   )

#   all_responses = [_response['choices'][i]['message']['content'] for i in range(len(_response['choices']))]
#   #filter non-int answers and converts to int
#   #filter untrustworthy answers
#   all_responses2= [ int(x) for x in all_responses if x.isdigit() and int(x) >= 0 and int(x) <= max_score ]
#   #average over the responses
#   if len(all_responses2) >=10:
#     res_avg = sum(all_responses2)/len(all_responses2)
#     # Return the score as a string
#     return str(res_avg)
#   else:
#     # Log the bad response
#     return 'BAD_RESPONSE'
      

# def run_gpt4_eval_vectorized(ResultsDF, Evalprompt, EvalCol, model='gpt-3.5-turbo-0301', max_score=5):
#     """Runs the GPT-4 model on the given data and returns the result as a DataFrame.
#     ResultsDF: The DataFrame containing the data to be evaluated
#     Evalprompt: The prompt that will be loaded for evaluation
#     EvalCol: The column of ResultsDF that will hold the results
#     model: The model to use for evaluation
#     max_score: The maximum score that can be given by the model (Used for filtering out bad responses)

#     Returns: A DataFrame containing the evaluation results
#     """
#     openai.api_key = OPENAI_API_KEY
#     prompt = open(Evalprompt).read()

#     # Apply the evaluate_row function to each row of the ResultsDF DataFrame
#     tqdm.tqdm.pandas()
#     ResultsDF[EvalCol] = ResultsDF.progress_apply(lambda row: evaluate_row(row, prompt, model, max_score), axis=1)

#     # Filter out the bad responses
#     bad_responses = ResultsDF[ResultsDF[EvalCol] == 'BAD_RESPONSE']

#     return bad_responses