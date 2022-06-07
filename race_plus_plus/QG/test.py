import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from keras.preprocessing.sequence import pad_sequences


MAXLEN_passage = 512
MAXLEN_question = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--model_path', type=str, help='Load path of trained model')
parser.add_argument('--num_beams', type=int, default=1, help='Number of beams')
parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of return sequences')
parser.add_argument('--test_data_path', type=str, help='Load path of test data')
parser.add_argument('--part_num', type=int, default=0, help='Indicate which fraction of data to generate samples on')
parser.add_argument('--save_path', type=str, help='Path to save generated text')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    return torch.device('cpu')
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    with open(args.test_data_path + "middle.json") as f:
        middle_data = json.load(f)
    with open(args.test_data_path + "high.json") as f:
        high_data = json.load(f)
    with open(args.test_data_path + "college.json") as f:
        college_data = json.load(f)
    test_data = middle_data + high_data + college_data    

    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    count = 0

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    all_generated_questions = []
    all_contexts = []

    NUM = 1500

    if args.part_num == 0:
        start = 0
        end = len(test_data)
    else:
        start = NUM * (args.part_num - 1)
        end = min( NUM * args.part_num , len(test_data) )

    for item in test_data[start:end]:
        context = item["article"]
        passage_encodings_dict = tokenizer(context, truncation=True, max_length=MAXLEN_passage, padding="max_length", return_tensors="pt")
        inp_id = passage_encodings_dict['input_ids']
        inp_att_msk = passage_encodings_dict['attention_mask']
        count+=1
        # if count==20:
        #     break
        print(count)

        all_generated_ids = model.generate(
            input_ids=inp_id,
            attention_mask=inp_att_msk,
            num_beams=args.num_beams, # Less variability
            # do_sample=True,
            # top_k=50,           # This parameter and the one below create more question variability but reduced quality of questions
            # top_p=0.95,          
            max_length=512,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            num_return_sequences=args.num_return_sequences
        )
        #print(len(all_generated_ids))
        for generated_ids in all_generated_ids:
            genQu = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            all_generated_questions.append(genQu)
            all_contexts.append(context.replace('\n', ' '))

    with open(args.save_path+str(args.part_num)+"_gen_questions.txt", 'w') as f:
        f.writelines("%s\n" % qu for qu in all_generated_questions)

    with open(args.save_path+str(args.part_num)+"_contexts.txt", 'w') as f:
        f.writelines("%s\n" % qu for qu in all_contexts)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)