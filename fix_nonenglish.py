import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-base')
from unidecode import unidecode

tokenizer = T5Tokenizer.from_pretrained('t5-base')
def extract_ent(fname1):
    d = json.load(open(fname1, 'r'))
    with open('g2t_ent.txt', 'w') as wf:
        for x in d:
            wf.write(' '.join([' '.join(xx) for xx in x['entities']])+'\n')
def recover(fname1, fname2):
    with open(fname1, 'r') as f1, open(fname2, 'r') as f2, open(fname1+'.fix', 'w') as wf:

        for str1, orig in zip(f1.readlines(), f2.readlines()):
            x1 = str1.split()
            x2 = orig.split()
            d = {}
            for x in x2:
                _x = x.replace('\u2013', '-').replace('~', '-').replace('`', "'").replace('\u2019', "'").replace('^', '')
                _x = unidecode(_x)
                _x = tokenizer.decode(tokenizer(_x)['input_ids'])
                if _x != x:
                    d[_x] = x
            r = []
            for x in x1:
                r.append(d.get(x, x))
            wf.write(' '.join(r)+'\n')
recover('hyp7.txt', 'g2t_ent.txt')
