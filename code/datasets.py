import pandas as pd
import torch
from torch.utils.data import Dataset
import pprint

# --------------------------------------------- SciAbstractsDataset ---------------------------------------------
    
class SciAbstractsDataset(Dataset):

  # ------------------------------------------------
  #  __init__
  #-------------------------------------------------
  def __init__(self, data, tasks, text_fields, label_fields, position_fields, add_distance_tokens, add_order_tokens, additional_features_tokens, instance_id_field_single, instance_id_field_pair, label_dict, max_length, tokenizer):
    # Reset indices in data. The column 'index' keeps a reference to the original indices.
    self.data = data.reset_index()
    self.tasks = tasks    
    self.text_fields = text_fields
    self.label_fields = label_fields
    self.position_fields = position_fields
    self.add_distance_tokens = add_distance_tokens   
    self.add_order_tokens = add_order_tokens       
    self.additional_features_tokens = additional_features_tokens    
    self.instance_id_field_single = instance_id_field_single
    self.instance_id_field_pair = instance_id_field_pair  
    self.label_dict = label_dict
    #self.max_length = max_length
    self.tokenizer = tokenizer
    self.predictions = []
  
  #def update_predictions(self, predictions):
  #  self.predictions = predictions
  
  #-------------------------------------------------
  # __len__
  #-------------------------------------------------  
  def __len__(self):
    return len(self.data)
    

  # The __getitem__ method of your dataset should return a dict whose keys should match the argument names for your models forward method.
  # For BERT see https://huggingface.co/transformers/model_doc/bert.html?highlight=berttokenizer#bertmodel
  # By default, BertTokenizer returns a dictionary with keys: 'input_ids', 'token_type_ids', 'attention_mask'.  
  #-------------------------------------------------
  # __getitem__
  #-------------------------------------------------  
  def __getitem__(self, index):
    texts = []
    # Only two texts supported by tokenizer - if more they should be concatenated manually.
    for i in range(len(self.text_fields)):
        texts.append(self.data.loc[index, self.text_fields[i]])

    # Tokens for relative positions and distance - for pairs of sentences.
    if len(texts) > 1:
        instance_id_field = self.instance_id_field_pair
        token_order = ''    
        token_distance = ''
        positions = []    
        if len(self.position_fields) == 2:
            positions.append(self.data.loc[index, self.position_fields[0]])     
            positions.append(self.data.loc[index, self.position_fields[1]])                    
            if (self.add_order_tokens):    
                token_order = '[BEFORE]' if positions[1] < positions[0] else '[AFTER]'  
            if (self.add_distance_tokens):
                distance = abs(positions[0]-positions[1])
                token_distance = '[DISTANCE_' + str(distance) + ']'
    else:
        instance_id_field = self.instance_id_field_single
            
    # Instance ID - to show predictions
    instance_id = self.data.loc[index, instance_id_field]

    # Tokens for additional features
    # E.g.: additional_features_tokens = {'doc_id': 'DOC', 'adu1_pos': 'POS', 'annotator': 'ANN'} 
    # Additional tokens for first sentence / second sentence respectively.
    additional_tokens = []
    for feature_field in sorted(self.additional_features_tokens):
        feature_token_prefix = self.additional_features_tokens[feature_field]
        feature_value = self.data.loc[index, feature_field]
        feature_token_value = str(feature_value).replace(' ', '_').upper()
        additional_tokens.append('[' + feature_token_prefix + '_' + feature_token_value + ']')
  
    label_ids = {}    
    for task in self.tasks:
        if (self.label_fields[task] in self.data):
            label_name_task = self.label_fields[task]
            label_name_item_task = self.data.loc[index, label_name_task]  
            label_id_item_task = self.label_dict[task][label_name_item_task]
            label_ids[task] = label_id_item_task
        else:
            label_ids[task] = None    
            
    # Encoding for one sentence / two sentences respectively.
    # Encodings for one sentence
    if len(texts) == 1:
        encoding = self.tokenizer(
          # Preprend additional tokens to sentence.
          text=' '.join(additional_tokens) + ' ' + texts[0],
          add_special_tokens=True,
          truncation=True,
          padding='max_length', # Padding criteria.
          return_attention_mask=True,
          return_tensors='pt',
        )
    else:
        # Encodings for two sentences
        encoding = self.tokenizer(
          # Preprend additional tokens to first sentence.
          text=token_order + ' ' + token_distance + ' ' + ' '.join(additional_tokens) + ' ' + texts[0],
          text_pair=texts[1],
          add_special_tokens=True,
          truncation=True,
          padding='max_length', # Padding criteria.
          return_attention_mask=True,
          return_tensors='pt',
        )
        
    # DEBUG
    if index == 0:
        print()
        print(f'__getitem__: instance_id - {instance_id}')
        print(self.tokenizer.decode(encoding['input_ids'].flatten())) 
        
    return {
      'instance_id': instance_id, 
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': label_ids,
    }  

    