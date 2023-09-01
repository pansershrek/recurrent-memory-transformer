### DECODER ###
import torch
import types
from modeling_rmt.language_modeling import MemoryCell
class MemoryCellGenerate(MemoryCell):
    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, write_mem=True, **kwargs)
        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(out, **kwargs)

        return out, new_memory_state
    
    def generate(self, input_ids, memory_state, attention_mask=None, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask, write_mem=False)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, write_mem, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        if write_mem:
            inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
        else:
            inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs
    
    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, self.num_mem_tokens: self.num_mem_tokens + attention_mask.shape[1]] = attention_mask
            return mask


from modeling_rmt.language_modeling import RecurrentWrapper

class RecurrentWrapperGenerate(RecurrentWrapper):
    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        memory_state = None

        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

        return out
    
# class RecurrentWrapperGenerate(RecurrentWrapper):
#     def generate(self, input_ids, input_ids_generate=None, attention_mask=None, **generate_kwargs):
#         memory_state = None

#         if input_ids_generate is not None:
#             input_ids = input_ids_generate
#         segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

#         for seg_num, segment in enumerate(segmented[:-1]):
#             cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)

#         final_segment = segmented[-1]
#         out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

#         return out
    

class RecurrentWrapperCustomForward(RecurrentWrapper):
    def __init__(self, *args, **rmt_kwargs):
        super().__init__(*args, **rmt_kwargs)
        
        base_model_forward = rmt_kwargs.get('base_model_forward_func', False)
        if not base_model_forward:
            raise ValueError("'base_model_forward_func' is undefined")
        self.override_base_model_forward(base_model_forward)
        self.memory_storage = {}

    def override_base_model_forward(self, custom_forward):
        new_forward = lambda *args, **kwargs: custom_forward(*args, **kwargs, rmt_parent=self)
        self.memory_cell.model.gpt_neox.forward = types.MethodType(new_forward, self.memory_cell.model.gpt_neox)


### ENCODER ###
###
import torch
from modeling_rmt.sequence_classification import *
from modeling_rmt.conditional_generation import *

class RMTEncoderCPUOffload(RMTEncoderForSequenceClassification):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)
        if self.num_mem_tokens == 0:
            segmented = segmented[-1:]

        base_model_outputs = []
        for seg_num, segment_input_ids in enumerate(segmented):                
            if self.rmt_config['bptt_depth'] != -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]
            # self.to_fwd_device(seg_kwargs)
            out = self.model(**seg_kwargs)
            base_model_outputs.append(out)
            base_model_outputs = base_model_outputs[-1:]

            
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

        out = self.process_outputs(base_model_outputs, output_attentions, output_hidden_states)
        return out
    

    # def to_fwd_device(self, kwargs):
    #     for k in kwargs:
    #         kwargs['k'] = kwargs['k'].to(self.rmt_config['fwd_device'])
    

class RMTEncoderMemFromSep(RMTEncoderForSequenceClassification):
    
    def extend_word_embeddings(self, num_mem_tokens, tokenizer):        
        vocab_size = self.model.config.vocab_size
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)

        special_tokens = tokenizer.special_tokens_map
        mem_start_ind = int('cls_token' in special_tokens or 'bos_token' in special_tokens)
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)
        self.model.embeddings = self.model.get_input_embeddings()

        self.reinit_memory_embeddings()

    def reinit_memory_embeddings(self):
        sep_embedding = self.model.embeddings.weight[self.sep_token][0]
        memory_weights = torch.stack([sep_embedding] * self.num_mem_tokens)
        noise_scale = self.model.embeddings.weight.std() / 10
        noise = torch.randn_like(memory_weights) * noise_scale
        self.model.embeddings.weight.data[self.memory_position] = memory_weights + noise


class RMTEncoderDecoderMemFromEos(RMTEncoderDecoderForConditionalGeneration):
   def extend_word_embeddings(self, num_mem_tokens, tokenizer):
            
        vocab_size = self.model.config.vocab_size
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)

        # fix scale and tie weights
        embeddings = self.model.get_input_embeddings()
        embeddings.weight.data[-num_mem_tokens:] = embeddings.weight.data[-num_mem_tokens:].normal_(mean=0.0, std=23.19373) / 2 + embeddings.weight.data[tokenizer.eos_token_id]
        self.model.set_input_embeddings(embeddings)
        self.model.tie_weights()
        # end

        special_tokens = tokenizer.special_tokens_map
        mem_start_ind = int('cls_token' in special_tokens or 'bos_token' in special_tokens)
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)
        self.model.embeddings = self.model.get_input_embeddings()
    

import torch
import torch.nn.functional as F
from modeling_rmt.base import RMTBaseModel
class RMTEncoderTBPTT(RMTEncoderForSequenceClassification):

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        init_memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)
        if self.num_mem_tokens == 0:
            segmented = segmented[-1:]

        memory_states = [(None, init_memory)]
        base_model_outputs = []
        for seg_num, segment_input_ids in enumerate(segmented):
            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) < 1:
                raise NotImplementedError
            
            memory = memory_states[-1][1].detach()
            memory.requires_grad = True
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]
            out = self.model(**seg_kwargs)
            base_model_outputs.append(out)
            
            self.memory_states = memory_states
            new_memory = out.hidden_states[-1][:, self.memory_position]
            memory_states.append((memory, new_memory))
        
        self.memory_states = memory_states

        out = self.process_outputs(base_model_outputs, output_attentions, output_hidden_states)
        return out
    
    def truncated_backward(self, k1, k2):
        memory_states = self.memory_states
        if k1 != -1:
            raise NotImplementedError
        
        for i in range(k2 - 1 if k2 != -1 else len(memory_states)):
            curr_grad = memory_states[-i-1][0].grad
            memory_states[-i-2][1].backward(curr_grad, retain_graph=False)

            # if we get all the way back to the "init_memory", stop
            if memory_states[-i-2][0] is None:
                break


