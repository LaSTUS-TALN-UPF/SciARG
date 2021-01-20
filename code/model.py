from transformers import BertModel, BertPreTrainedModel
import torch
from torch import nn
from torch import cat
from torch.nn import CrossEntropyLoss, MSELoss
import pprint
import json
import os.path

class BertWithFeatForSequenceClass(BertPreTrainedModel):
    def __init__(self, config, *model_args, **kwargs):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.saved_path = kwargs['saved_path'] if 'saved_path' in kwargs else '' 
        # These attributes are overriden if the model is loaded.
        self.tasks = kwargs['tasks'] if 'tasks' in kwargs else []
        self.pooled_tokens = kwargs['pooled_tokens'] if 'pooled_tokens' in kwargs else []
        self.label_dict = kwargs['label_dict'] if 'label_dict' in kwargs else {}   
        self.classifiers = {}
        if self.saved_path:
            # Loaded model
            self.init_additional(self.saved_path)
            # The classifiers within a dictionary are not loaded automatically.
            self.load_classifiers(self.saved_path)            
        else:
            # New model
            for task in self.tasks:
                num_labels_task = len(self.label_dict[task])         
                self.classifiers[task] = nn.Linear(self.config.hidden_size*(1+len(self.pooled_tokens)), num_labels_task)
                self.classifiers[task].apply(self._init_weights)                    
        self.pooler_dense = nn.Linear(self.config.hidden_size*(1+len(self.pooled_tokens)), self.config.hidden_size)                
        self.pooler_activation = nn.Tanh()                
        # Weights are overriden with saved weights if model created with from_pretrained. 
        self.init_weights()  

    @classmethod
    def load_model(cls, model_path, *model_args, **kwargs):
        kwargs['saved_path'] = model_path
        model = cls.from_pretrained(model_path, *model_args, **kwargs)
        return model      

    def init_additional(self, save_directory):
        additional_config_file = save_directory + '/additional_config.json'
        if os.path.isfile(additional_config_file):    
            with open(additional_config_file, 'r') as fp:
                additional_config = json.load(fp)
            self.pooled_tokens = additional_config['pooled_tokens']                
        label_dict_file = save_directory + '/label_dict.json'
        if os.path.isfile(additional_config_file):            
            with open(label_dict_file, 'r') as fp:
                self.label_dict = json.load(fp)
        self.tasks = self.label_dict.keys()
        
    def get_parameters(self):
        params = [p for p in self.parameters()]
        for task in self.tasks:
            params.extend([p for p in self.classifiers[task].parameters()])
        return params
                 
    def load_classifiers(self, save_directory):
        for task in self.tasks:
            path_classifier = save_directory + '/' + task + '.pt'   
            num_labels_task = len(self.label_dict[task])         
            self.classifiers[task] = nn.Linear(self.config.hidden_size*(1+len(self.pooled_tokens)), num_labels_task)            
            self.classifiers[task].load_state_dict(torch.load(path_classifier))
            self.classifiers[task].eval()        

    def save_model(self, save_directory):
        self.save_pretrained(save_directory=save_directory)    
        additional_config_dict = {'pooled_tokens': self.pooled_tokens}
        with open(save_directory + '/additional_config.json', 'w') as fp:
            json.dump(additional_config_dict, fp)         
        with open(save_directory + '/label_dict.json', 'w') as fp:
            json.dump(self.label_dict, fp)        
        for task in self.classifiers:
            path_classifier = save_directory + '/' + task + '.pt'
            torch.save(self.classifiers[task].state_dict(), path_classifier)
        
    def to_device(self, device):
        for task in self.tasks:
            self.classifiers[task] = self.classifiers[task].to(device)
        self = self.to(device)   

    def forward(
        self,
        task=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Pooling
        if self.pooled_tokens:
            # TODO: Generalize for multi-task.
            token_representations = outputs[0]      
            cls_representation = token_representations[:,0,:]  
            pooled_representations = cls_representation
            for token_position in self.pooled_tokens:
                feature_representation = token_representations[:,token_position,:]       
                pooled_representations = cat((pooled_representations, feature_representation), 1)
            # Taken from BertPooler but using the concatenation of all tokens representations.
            pooled_representations = self.pooler_dense(pooled_representations)
            pooled_output = self.pooler_activation(pooled_representations)
            pooled_output = pooled_representations
        else:
            # CLS token representation is returned in outputs[1] - already passed through a dense layer with tanh activation.        
            pooled_output = outputs[1]       
        pooled_output = self.dropout(pooled_output)
        num_labels_task = len(self.label_dict[task])
        logits = self.classifiers[task](pooled_output)
        if len(labels) > 0:
            if num_labels_task == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels_task), labels.view(-1))

        if not return_dict:
            output = (logits, loss,) + outputs[2:]
            return output

        return {
            'logits': logits,
            'loss': loss,            
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
