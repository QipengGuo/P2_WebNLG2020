import tqdm
import random
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import json
from sklearn.metrics import f1_score
from collections import defaultdict
import copy
from transformers import T5Tokenizer, T5ForConditionalGeneration
from unidecode import unidecode
from sacremoses import MosesTokenizer, MosesDetokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-large')
import logging
logging.basicConfig(level=logging.INFO)
logging.info('Start Logging')

def jload(fname):
    t = open(fname,'r').read().replace('\u2013', '-').replace('~', '-').replace('`', "'").replace('\u2019', "'").replace('^', '')
    return unidecode(t).split('\n')[:-1]

def prep_data(config, fname):
    #prep data always has two steps, build the vocabulary first and then generate data samples
    nsplit = int(config['gpus'])
    train_raw = jload(config['train_file'])
    train = [('Graph to Text: ' + x.split('||')[0].replace('_', ' '), x.split('||')[1]+' </s>') for x in train_raw]
    dev_raw = jload(config['dev_file'])
    dev = [('Graph to Text: ' + x.split('||')[0].replace('_', ' '), x.split('||')[1]+' </s>') for x in dev_raw]
    test_raw = jload(config['test_file'])
    test = [('Graph to Text: ' + x.replace('_', ' '), None) for x in test_raw]
    print(len(train), len(dev), len(test))
    sp = len(train)//nsplit

    for i in range(nsplit):
        torch.save({'train':train[i*sp:(i+1)*sp], 'dev':dev, 'test':test}, fname+str(i))

def pred_one(batch, model):
    model.eval()
    inp = tokenizer([x[0] for x in batch], return_tensors='pt', padding=True)['input_ids'].to(model.device)

    pred = model.generate(input_ids=inp, max_len=50, num_beams=4)
    return tokenizer.batch_decode(pred)

def train_g2t_one_step(batch, model, optimizer, config):
    model.train()
    optimizer.zero_grad()
    inp = tokenizer([x[0] for x in batch], return_tensors='pt', padding=True)['input_ids'].to(model.device)
    tar = tokenizer([x[1] for x in batch], return_tensors='pt', padding=True)['input_ids'].to(model.device)

    loss = model(input_ids=inp, labels=tar)[0]

    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= config['gpus']
    nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
    optimizer.step()
    return loss.item()


def eval_g2t(datas, model, demo_name='hyp.txt'):
    model.eval()
    hyp = []
    with tqdm.tqdm(batch_it(datas, 2)) as tqb:
        for i, batch in enumerate(tqb):
            with torch.no_grad():
                pred = pred_one(batch, model)
            hyp.extend(pred)

    mt = MosesTokenizer(lang='en')
    md = MosesDetokenizer(lang='en')
    wf_h = open(demo_name, 'w')
    for i,h in enumerate(hyp):
        wf_h.write(md.detokenize(mt.tokenize(str(h)))+'\n')
    wf_h.close()

    return 0.0

def batch_it(datas, batch_size):
    r = []
    ret = []
    for x in datas:
        r.append(x)
        if len(r)==batch_size:
            ret.append(r)
            r = []
    if len(r)>0:
        ret.append(r)
    return ret 

def train(proc_id, devices, _type, config, fname='tmp_data.pt'):
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dev_id = devices[proc_id]
    device = torch.device(dev_id)
    _d = torch.load(fname+str(proc_id))
    train_d = _d['train']
    dev_d = _d['dev']
    test_d = _d['test']

    port = 12346
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port=str(port))

    torch.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=len(devices),
                                          rank=dev_id)


    model = T5ForConditionalGeneration.from_pretrained('t5-large')
    model.config.update(model.config.task_specific_params['translation_en_to_de'])
    model.to(device)
   

    from transformers.optimization import get_cosine_schedule_with_warmup , get_linear_schedule_with_warmup	
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], weight_decay=config['weight_decay']) 
#    schedulerG2T = get_cosine_schedule_with_warmup(
#		optimizer = optimizerG2T , 
#		num_warmup_steps = 1500 , 
#		num_training_steps = 3000 * config['main']['epoch'], 
#	)
    
    losses = []
    for i in range(0, config['epoch']):
        with tqdm.tqdm(batch_it(train_d, config['batch_size'])) as tqb:
            for j, batch in enumerate(tqb):
                loss = train_g2t_one_step(batch, model, optimizer, config)
                losses.append(loss)
                tqb.set_postfix({'loss': np.mean(losses)})


        logging.info('Epoch '+str(i))

        if i%1==0 and proc_id==0:
            torch.save(model.state_dict(), config['save']+'X'+str(i))
            eval_g2t(test_d, model, 'hyp'+str(i)+'.txt') 
        torch.distributed.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))
    _config = copy.deepcopy(config)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fname = 'tmp_data.pt'
    prep_data(config, fname)

    devices = list(range(config['gpus']))

    torch.multiprocessing.spawn(train, args=(devices, 'train', config, fname), nprocs=len(devices))

if __name__=='__main__':
    main()

