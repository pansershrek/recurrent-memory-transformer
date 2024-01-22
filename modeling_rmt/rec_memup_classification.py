import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import transformers
import random


class MemUPModule(torch.nn.Module):

    def __init__(self, rnn_core, predictor, num_mem_tokens):
        super().__init__()
        self.rnn_core = rnn_core
        self.predictor = predictor
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        model = self.rnn_core
        embeddings = model.get_input_embeddings()
        memory_dim = getattr(model.config, 'n_embd', model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        #adds memory states as prefix and suffix and fixes masks accordingly
        seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
        predictor_mode = seg_kwargs.pop("predictor_mode", None)
        if predictor_mode:
            out = self.predictor(**seg_kwargs)
        else:
            out = self.rnn_core(**seg_kwargs)
        #memory state is taken from suffix and output is stripped from memory states
        out, new_memory_state = self.process_output(out, **kwargs)

        return out, new_memory_state

    def generate(self, input_ids, memory_state, attention_mask, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'],
                                  attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, **kwargs):
        seg_kwargs = dict(**kwargs)
        model = self.rnn_core
        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = model.get_input_embeddings()(input_ids)
        if self.num_mem_tokens > 0:
            inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)

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
            mask[:, self.num_mem_tokens:-self.num_mem_tokens] = attention_mask
            return mask

    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
            out['logits'] = model_outputs.logits[:, self.num_mem_tokens:-self.num_mem_tokens]

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, self.num_mem_tokens:-self.num_mem_tokens] for lh in
                                        model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state


class RecurrentMemUP(torch.nn.Module):

    def __init__(self, recurrent_memory, **rmt_kwargs):
        super().__init__()
        self.recurrent_memory = recurrent_memory
        self.rmt_config = rmt_kwargs

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None,
                output_attentions=None, output_hidden_states=None):

        input_segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        labels_segmented = self.segment(labels_mask=labels_mask, labels=labels)

        rollout = self.rmt_config.get("k2", -1)
        pred_freq = self.rmt_config.get("pred_freq", 2)
        if rollout > 0:
            pred_freq = min(pred_freq, rollout)


        trg_segment_inputs, trg_segment_labels = self.select_target_segments(input_segmented, labels_segmented)

        pred_outputs = []
        memory_state = None
        last_seg = len(input_segmented) - 1
        i = 0

        while True:
            # if rollout is limited AND current rollout reached it's max length:
            is_pred_step = self.training and (i % pred_freq == 0) and (i > 0)
            is_last_step = i == last_seg
            if is_pred_step or is_last_step:
                # make prediction about target segment using current state of the memory
                pred_out, _ = self.recurrent_memory(
                    **trg_segment_inputs,
                    memory_state=memory_state,
                    predictor_mode=True,
                    output_hidden_states=True
                )
                pred_outputs.append(pred_out)

            if i >= last_seg: break

            memory_state = self.manage_mem_grads(memory_state, i)

            rnn_out, memory_state = self.recurrent_memory(
                **input_segmented[i], memory_state=memory_state, output_hidden_states=True
            )
            i += 1

        if self.training is False:
            if len(pred_outputs) != 1:
                raise ValueError("We don't need intermedate predictions during evaluation!")

        # if len(pred_outputs) == 0:
        #     self.investigate_problem(input_ids, attention_mask, input_segmented, pred_outputs, rollout, pred_freq)
        out = self.make_prediction(pred_outputs,
                                   trg_segment_labels=trg_segment_labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)

        return out

    # def investigate_problem(self, input_ids, attention_mask, segmented, pred_outputs, rollout, pred_freq):
    #     print("INVESTIGATION:")
    #     print(f"rollout={rollout}, pred_freq={pred_freq}")
    #     print("sample size:", attention_mask.sum(-1))
    #     print("size of segments:", [s['input_ids'].size(-1) for s in segmented])
    #     print(f"number of pred_outputs: {len(pred_outputs)}")
    #     #raise NotImplementedError()

    def select_target_segments(self, *args):
        #input_segment_trg = input_segments[-1]
        #label_segment_trg = label_segments[-1]
        # assert all([s['labels_mask'].sum() == 0 for s in  label_segments]), "sequences are not aligned right"
        #return input_segment_trg, label_segment_trg
        return tuple(a[-1] for a in args)

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        raise NotImplementedError("RecurrentMemUP.generate")
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out, memory_state = self.recurrent_memory(**segment, memory_state=memory_state, output_hidden_states=True)

        final_segment = segmented[-1]
        out = self.recurrent_memory.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

        return out

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments

    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment')
        segment_size = self.rmt_config.get('segment_size')
        if align in {'left', None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align in {'right', None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def make_prediction(self, cell_outputs, trg_segment_labels, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=0)
        full_hidden_states = tuple(
             [torch.cat(layer_hs, dim=0) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

        labels = trg_segment_labels.get("labels", None)
        #labels = kwargs.get('labels')
        if labels is not None:
            labels = labels.repeat(len(cell_outputs), 1)
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))

            loss_fct = CrossEntropyLoss()
            labels_mask = trg_segment_labels.get('labels_mask')
            if labels_mask is not None:
                labels_mask = labels_mask.repeat(len(cell_outputs), 1)
                shift_mask = labels_mask[..., :-1].contiguous()
                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]

            out['loss'] = loss_fct(flat_logits, flat_labels)
            if out['loss'] is None:
                raise ValueError
        else:
            out['loss'] = 0

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f'{key}_{seg_num}'] = value

        return out

    def manage_mem_grads(self, memory_state, seg_num):
        r = self.rmt_config.get("k2", -1)
        #if Trancation is True AND rollout is ended AND mem is not None
        if r > 0 and seg_num % r == 0 and memory_state is not None:
            return memory_state.detach()