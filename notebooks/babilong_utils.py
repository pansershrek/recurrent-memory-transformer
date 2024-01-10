import pandas as pd
import numpy as np
import re
import nltk
from torch.utils.data import Dataset

# preprocess babi text files
def get_dataset_df(dataset_path):
    with open(dataset_path, 'r') as f:
        texts = f.read().strip()
        texts = texts.split('\n')
        df = pd.DataFrame(texts, columns=['text'])

    # parse samples
    df['phrase_num'] = df.text.apply(lambda x: int(x.split(' ')[0]))
    df.text = df.text.apply(lambda x: x[x.index(' ') + 1:])
    df['answer'] = df.text.apply(lambda x: x[x.index('\t') + 1:] if '\t' in x else None)
    # df['reference_num'] = df.answer.apply(lambda x: x if x is None else x.split('\t| ')[1:])
    df['reference_num'] = df.answer.apply(lambda x: x if x is None else [int(n) for n in re.split('\t| ', x)[1:]])
    df.answer = df.answer.apply(lambda x: x if x is None else x.split('\t')[0])
    df.text = df.text.apply(lambda x: x.split('\t')[0] if '\t' in x else x)

    # mark each sample
    sample_start_inds = list(np.where(df.phrase_num == 1)[0]) + [df.shape[0]]
    for i, (start, end) in enumerate(zip(sample_start_inds, sample_start_inds[1:])):
        df.loc[start:end, 'initial_sample_num'] = i

    df.initial_sample_num = df.initial_sample_num.astype(int)

    # multiple questions in sample -> samples with single question
    initial_samples = [df[df.initial_sample_num == sn] for sn in df.initial_sample_num.unique()]

    single_question_slices = []
    for sample in initial_samples:
        answer_positions = sample[~sample.answer.isna()].index
        slices = [sample[:ans_pos+1] for ans_pos in answer_positions]
        for i, slc in enumerate(slices):
            slices[i] = slc[(slc.answer.isna()) | (slc.index == slc.index[-1])]
        single_question_slices += slices
    
    df = pd.concat(single_question_slices).reset_index(drop=True)

    # mark each sample again
    sample_start_inds = list(np.where(df.phrase_num == 1)[0]) + [df.shape[0]]
    for i, (start, end) in enumerate(zip(sample_start_inds, sample_start_inds[1:])):
        df.loc[start:end, 'sample_num'] = i

    df.sample_num = df.sample_num.astype(int)
    
    return df


# babi task loader dataset
class TaskDataset(Dataset):
    def __init__(self, dataset_path):
        self.fact_dataset = get_dataset_df(dataset_path)

    def __getitem__(self, ind):
        slc = self.fact_dataset[self.fact_dataset.sample_num == ind]
        references = slc[slc.phrase_num.isin(slc.reference_num.values[-1])].text.values
        sample = {'facts': slc.text.values[:-1],
                  'question': slc.text.values[-1],
                  'answer': slc.answer.values[-1],
                  'references': references}
        return sample
    
    def __len__(self):
        return self.fact_dataset.sample_num.max()
    


def sum_lengths(sentences):
    return sum([len(s) for s in sentences])


# sampler of background text 
class SentenceSampler:
    def __init__(self, dataset, tokenizer):
        self.sample_ind = 0
        self.dataset = dataset
        self.sentences = []
        self.tokenizer = tokenizer
        self.sentence_tokenizer = nltk.PunktSentenceTokenizer()

    def get_sample(self, sample_size):
        total_tokens = sum_lengths(self.sentences)
        while total_tokens < sample_size: # add a new dataset item
            text = self.dataset[self.sample_ind]['text']
            self.sample_ind += 1
            sentences = self.sentence_tokenizer.tokenize(text)
            tokenized = [self.tokenizer.encode(s, add_special_tokens=False) for s in sentences]
            self.sentences += tokenized
            total_tokens += sum_lengths(tokenized)

        sample = []
        sample_tokens = 0
        for sent in self.sentences: # add new sentence until sample_size is reached
            sample_tokens += len(sent)
            if sample_tokens >= sample_size:
                break
            sample.append(sent)
            self.sentences = self.sentences[1:]
        
        return sample
    

# combined dataset for noisy babi QA
class NoiseInjectionDataset(Dataset):
    def __init__(self, task_dataset, noise_sampler, tokenizer, sample_size=100):
        self.task_dataset = task_dataset
        self.noise_sampler = noise_sampler
        self.sample_size = sample_size
        self.tokenizer = tokenizer

    def __getitem__(self, ind):
        sample = self.task_dataset[ind]
        facts_tok = self.tokenizer(list(sample['facts']))['input_ids']
        question_tok = self.tokenizer(sample['question'])['input_ids']
        answer_tok = self.tokenizer(sample['answer'])['input_ids']
        
        task_len = sum_lengths(facts_tok) + len(question_tok) + len(answer_tok)
        background_text_len = self.sample_size - task_len
        background_text = self.noise_sampler.get_sample(background_text_len)
        sample['background_text'] = background_text

        possible_positions = range(len(background_text) + 1) 
        fact_positions = np.random.choice(possible_positions, len(facts_tok))
        fact_positions.sort()
        sample['fact_positions'] = fact_positions       # positions of facts between noise sentences
        updated_sample = [[] for _ in range(len(background_text) + 1)] 
        for fact, pos in zip(facts_tok, fact_positions):
            updated_sample[pos].append(fact)

        updated_sample[-1].append(question_tok)

        for i, s in enumerate(background_text):
            updated_sample[i].append(s)

        flat = [i for s in updated_sample for i in s]
        tokens = [i for s in flat for i in s]

        sample['input_tokens'] = tokens
        sample['target_tokens'] = answer_tok

        return sample
    
    def __len__(self):
        return self.task_dataset
