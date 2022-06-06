"""
This file takes a set of questions (and corresponding contexts) and only retains those
that have 4 unique answer options
and accurate (i.e. args.filter_rate QA models agree on the first option as the correct answer)
"""

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy.lib.arraysetops import unique
from scipy.special import softmax

from transformers import ElectraTokenizer, ElectraForMultipleChoice, ElectraConfig
from keras.preprocessing.sequence import pad_sequences

MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--questions_path', type=str,  help='Specify path to generated questions')
parser.add_argument('--contexts_path', type=str,  help='Specify path to contexts')
parser.add_argument('--models_dir', type=str, help='Specify path to directory containing all trained QA models')
parser.add_argument('--save_dir', type=str, help='Specify path to save filtered questions')
parser.add_argument('--batch_size', type=int, default=4, help='Specify the training batch size')
parser.add_argument('--filter_rate', type=int, default=3, help='Minimum number of QA models that must agree out of 3')


def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def organise_data(questions, contexts):
    organised_data = []
    for question, context in zip(questions, contexts):
        first_sep_pos = question.find("[SEP]")
        qu = question[:first_sep_pos]
        opts = []
        validSEP = True
        sep_pos = first_sep_pos
        while validSEP:
            question = question[sep_pos+6:]
            sep_pos = question.find("[SEP]")
            if sep_pos == -1:
                validSEP = False
                opt = question
            else:
                opt = question[:sep_pos]
            opts.append(opt)
        curr_point = {'question': qu, 'context': context, 'options':opts, 'label':0}
        # print(curr_point)
        organised_data.append(curr_point)
    return organised_data

def got_four_opts(test_data):
    to_keep = []
    num_valid = 0
    for ex in test_data:
        unique_opts = []
        for opt in ex['options']:
            if opt not in unique_opts:
                unique_opts.append(opt)
        if len(unique_opts) == 4:
            to_keep.append(ex)
    return to_keep
        

def get_qa_predictions(test_data, models, device, args):

    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    input_ids = []
    token_type_ids = []
    count = 0
    for ex in test_data:
        question, context, options = ex['question'], ex['context'], ex['options']
        four_inp_ids = []
        four_tok_type_ids = []
        for opt in options:
            combo = context + " [SEP] " + question + " " + opt
            inp_ids = tokenizer.encode(combo)
            if len(inp_ids)>512:
                inp_ids = inp_ids[-512:]
            tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]
            four_inp_ids.append(inp_ids)
            four_tok_type_ids.append(tok_type_ids)
        four_inp_ids = pad_sequences(four_inp_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
        four_tok_type_ids = pad_sequences(four_tok_type_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
        input_ids.append(four_inp_ids)
        token_type_ids.append(four_tok_type_ids)

    # Create attention masks
    attention_masks = []
    for sen in input_ids:
        sen_attention_masks = []
        for opt in sen:
            att_mask = [int(token_id > 0) for token_id in opt]
            sen_attention_masks.append(att_mask)
        attention_masks.append(sen_attention_masks)
    # Convert to torch tensors
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)

    ds = TensorDataset(input_ids, token_type_ids, attention_masks)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    logits_all_models = []
    for i, model in enumerate(models):
        print("Model:", i)
        logits = []
        count = 0
        for inp_id, tok_typ_id, att_msk in dl:
            print(count)
            count+=1
            inp_id, tok_typ_id, att_msk = inp_id.to(device), tok_typ_id.to(device), att_msk.to(device)
            with torch.no_grad():
                outputs = model(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id)
            curr_logits = outputs[0]
            logits += curr_logits.detach().cpu().numpy().tolist()
        logits_all_models.append(logits)
    logits_all_models = np.asarray(logits_all_models)
    return logits_all_models

def get_agreement(all_logits, filter_rate):
    isValid = []
    preds = []
    for seed_logits in all_logits:
        class_preds = np.argmax(seed_logits, axis=-1)
        preds.append(class_preds)
    for i in range(len(class_preds)):
        correct = 0
        for seed in range(len(all_logits)):
            pred = preds[seed][i]
            if pred==0:
                correct += 1
        if correct >= filter_rate:
            valid = True
        else:
            valid = False
        isValid.append(valid)
    return isValid

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    with open(args.questions_path, 'r') as f:
        all_gen_questions = [a.rstrip() for a in f.readlines()]

    with open(args.contexts_path, 'r') as f:
        all_contexts = [a.rstrip() for a in f.readlines()]

    organised_data = organise_data(all_gen_questions, all_contexts)

    # Remove examples that don't have 4 unique options
    filtered_data = got_four_opts(organised_data)

    device = get_default_device()
    models = []
    seeds = [1, 2, 3]
    for seed in seeds:
        model_path = args.models_dir + str(seed) + '/electra_QA_MC_seed' + str(seed) + '.pt'
        model = torch.load(model_path, map_location=device)
        model.eval().to(device)
        models.append(model)
    
    all_logits = get_qa_predictions(filtered_data, models, device, args)

    isValid = get_agreement(all_logits, args.filter_rate)
    final_filtered_data = []
    for sample, keep in zip(filtered_data, isValid):
        if keep:
            final_filtered_data.append(sample)

    # Save the filtered data
    with open(args.save_dir+'filtered.json', 'w') as f:
        json.dump(final_filtered_data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)