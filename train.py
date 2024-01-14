from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor
from IPython.display import HTML, display
import random
import io

print("train.py: savepath is hardcoded to ./")
savepath = "./"

# A progress bar because we can
def progress(loss,value, max=100):
    return HTML(""" Batch loss :{loss}
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(loss=loss,value=value, max=max))

# Designed to finetune a T5 model
# Modifies model directly
# Returns loss values for plotting
# dev: usually "cuda:0" 
def finetuneT5 (dev, data_train, model, tokenizer, num_of_epochs=5, batch_size=2):
  optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
  )

  num_of_batches=int(len(data_train)/batch_size)
  num_of_batches

  #Sets the module in training mode
  model.train()

  loss_per_10_steps=[]
  for epoch in range(1,num_of_epochs+1):
    print('Running epoch: {}'.format(epoch))

    running_loss=0

    out = display(progress(1, num_of_batches+1), display_id=True)
    for i in range(num_of_batches):
      inputbatch=[]
      labelbatch=[]
      new_df=data_train[i*batch_size:i*batch_size+batch_size]
      for indx,row in new_df.iterrows():
        input = 'Data: '+row['triples_input']+'</s>'
        labels = row['description_en']+'</s>'
        inputbatch.append(input)
        labelbatch.append(labels)
      inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,truncation=True,max_length=200,return_tensors='pt')["input_ids"]
      labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,truncation=True,max_length=200,return_tensors="pt") ["input_ids"]
      inputbatch=inputbatch.to(dev)
      labelbatch=labelbatch.to(dev)

      # clear out the gradients of all Variables
      optimizer.zero_grad()

      # Forward propogation
      outputs = model(input_ids=inputbatch, labels=labelbatch)
      loss = outputs.loss
      loss_num=loss.item()
      logits = outputs.logits
      running_loss+=loss_num
      if i%10 ==0:
        loss_per_10_steps.append(loss_num)
      out.update(progress(loss_num,i, num_of_batches+1))

      # calculating the gradients
      loss.backward()

      #updating the params
      optimizer.step()

    running_loss=running_loss/int(num_of_batches)
    print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))
  return loss_per_10_steps



########################################################
# Prompt handling functions
# Because I didn't use langchain from the beginning

def generate_prompt(row):
  """Original version, kept for compatibility"""
  return """Write a short description for this band in the first person. Here are facts provided as triplets: """ + row.triples_input

# REMIND: generic prompt
# List of basic prompt blocs. [0:8] contain basic blocs, [8:13] contain alternative beginnings, [13:18] contain alternative endings, 
# [18:23] contain alternative endings for the alternative beginnings, [22:27] contain aadditions that can be appendend.
Generic_Prompt1 = open(savepath + "Prompts/task1_zeroshot.txt", "r").read().split(";")
Few_Prompt1 = open(savepath + "Prompts/task1_fewshot.txt", "r").read().split(";")
Generic_Prompt2 = open(savepath + "Prompts/task2_zeroshot.txt", "r").read().split(";")

def generate_prompt_new(row, task1_in):
  """Generates a prompt for a given row of a dataframe"""
  prompt = ''.join(Generic_Prompt1[0:7]).replace('{{Name}}', task1_in.loc[row, 'f_name'])
  prompt = prompt.replace('{{Location}}', task1_in.loc[row, 'f_locality'])
  prompt = prompt.replace('{{Genres}}', task1_in.loc[row, 'f_genres'])
  prompt = prompt.replace('{{Events}}', task1_in.loc[row, 'f_eventtypes'])
  prompt = prompt.replace('{{Type}}', task1_in.loc[row, 'f_casts'])
  prompt = prompt.replace('{{Radius}}', str(task1_in.loc[row, 'f_distance']))
  return prompt

def gen_prompt1(row):
  """Like generate_prompt_new but pass it a single row of a dataframe"""
  prompt = ''.join(Generic_Prompt1[0:7]).replace('{{Name}}', row.f_name)
  prompt = prompt.replace('{{Location}}', row.f_locality)
  prompt = prompt.replace('{{Genres}}', row.f_genres)
  prompt = prompt.replace('{{Events}}', row.f_eventtypes)
  prompt = prompt.replace('{{Type}}', row.f_casts)
  prompt = prompt.replace('{{Radius}}', str(row.f_distance))
  return prompt

def shuffler(data):
  """Given a string of data, shuffles the words and returns a new string"""
  data = data.split(',')
  random.shuffle(data)
  return ', '.join(data)

def gen_prompt1_rand(row):
  """Like generate_prompt_new but pass it a single row of a dataframe.
     Randomizes the order of the facts in the prompt.
     Randomizes the order of categories inside some of the facts.
     Doesn't use the alternative beginnings and endings.
  """
  order = random.sample([1,2,3,4,5,6],6)
  prompt = ''.join([Generic_Prompt1[i] for i in [0] + order + [7]])
  prompt = prompt.replace('{{Name}}', row.f_name)
  prompt = prompt.replace('{{Location}}', row.f_locality)
  prompt = prompt.replace('{{Genres}}', shuffler(row.f_genres))
  prompt = prompt.replace('{{Events}}', shuffler(row.f_eventtypes))
  prompt = prompt.replace('{{Type}}', row.f_casts)
  prompt = prompt.replace('{{Radius}}', str(row.f_distance))
  return prompt

def gen_prompt1_alt(row):
  """Like generate_prompt_new but pass it a single row of a dataframe.
     Uses alternative beginnings/endings for the prompt.
     Randomizes the order of categories inside some of the facts.
     Makes use of some additions at the end of the prompt.
     Doens't randomize the order of the facts in the prompt.
  """
  alt_begin = random.choice([0,9,10,11]) # 12 excluded as they don't exist yet
  alt_end = random.choice([7,14,15,16]) # 17 excluded as it doesn't exist yet
  addition_end = random.choice([24, 27, 50,51,52,53]) #excluded 25, 26 as they are specific to german.
  promptorder = [alt_begin] + [1,2,3,4,5] + [alt_end]
  if addition_end < 50:
    promptorder.append(addition_end)
  prompt = ''.join([Generic_Prompt1[i] for i in promptorder])
  prompt = prompt.replace('{{Name}}', row.f_name)
  prompt = prompt.replace('{{Location}}', row.f_locality)
  prompt = prompt.replace('{{Genres}}', shuffler(row.f_genres))
  prompt = prompt.replace('{{Events}}', shuffler(row.f_eventtypes))
  prompt = prompt.replace('{{Type}}', row.f_casts)
  prompt = prompt.replace('{{Radius}}', str(row.f_distance))
  return prompt

def gen_prompt1_few(row, sample):
  """Like generate_prompt_new but pass it a single row of a dataframe
     Uses fewshot, so an example description is given
  """
  prompt = ''.join(Few_Prompt1[0:9]).replace('{{Name}}', row.f_name)
  prompt = prompt.replace('{{Description}}', sample)
  prompt = prompt.replace('{{Location}}', row.f_locality)
  prompt = prompt.replace('{{Genres}}', row.f_genres)
  prompt = prompt.replace('{{Events}}', row.f_eventtypes)
  prompt = prompt.replace('{{Type}}', row.f_casts)
  prompt = prompt.replace('{{Radius}}', str(row.f_distance))
  return prompt

def gen_prompt2(row, rel_gigs):
  """Like generate_prompt_new but pass it a single row of a dataframe.
  Adds information about the recent gigs."""
  #Convert gigs DF to string
  output = io.StringIO()
  rel_gigs.to_csv(output, sep=';', index=False)
  gigs_string = output.getvalue()

  prompt = ''.join(Generic_Prompt2[0:7]).replace('{{Name}}', row.f_name)
  prompt = prompt.replace('{{Location}}', row.f_locality)
  prompt = prompt.replace('{{Genres}}', row.f_genres)
  prompt = prompt.replace('{{Events}}', row.f_eventtypes)
  prompt = prompt.replace('{{Type}}', row.f_casts)
  prompt = prompt.replace('{{Gigs}}', str(gigs_string))
  return prompt

def gen_prompt3(rel_gigs, region):
  #Convert gigs DF to string
  output = io.StringIO()
  rel_gigs.to_csv(output, sep=';', index=False)
  gigs_string = output.getvalue()
  
  prompt = """Use the data about recent events and concerts in {{region}} to generate a regional summary. 
  Prioritize recent events and abstract from the data to create a summary of the region's music scene. 
  Mention planned events or popular locations and bands.
  The summary should be under 400 tokens."""
  prompt = prompt.replace('{{region}}', region)
  return gigs_string + prompt