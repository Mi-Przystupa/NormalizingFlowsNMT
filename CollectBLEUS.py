#!/usr/bin/env python
# coding: utf-8

# In[9]:


import argparse
import json
import torch
import glob
import os


# In[4]:


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='.results/tabular_en-de/', type=str,
    help="name of language direction dictory, make sure to end name with /")
parser.add_argument('--savedir', default='.results/collected_exp/', type=str,
    help="name of the save directory to store collected results")
parser.add_argument('--dataset', default='validate', type=str,
    help="dataset to load results from...if available")
parser.add_argument('--params', default=['model_type', 'kl_anneal', 'z_dim'])

args = parser.parse_args()


# In[11]:

outdir = args.outdir#'.results/tabular_en-de/'
params = args.params#['model_type', 'kl_anneal', 'z_dim']
save_dir = args.savedir

# In[22]:


def getDirs(path):
    return glob.glob(path + '*')

def getBLEUS(pths):
    err = 0
    ret = {}
    for p in pths:
        exp = p.split('/')[-1]
        print(p)
        try:
            with open(p + '/bleus.json', 'r') as f:
                bleus = json.load(f)
                ret[exp] = bleus
        except Exception as e:
            err += 1
    
    print("{}/ {} did not have BLEUS".format(err, len(pths)))
    return ret
            
def getPerplexities(pths):
    err = 0
    ret = {}
    for p in pths:
        exp = p.split('/')[-1]
        found_perplex = True
        try:
            perplexities = torch.load(p + '/perplexities.pth')
        except Exception as e:
            found_perplex = False
        found_args = True
        try:
            with open(p + '/arguments.json', 'r') as f:
                args = json.load(f)
        except Exception as e:
            found_args =False
            print(e)
        if found_perplex and found_args:
            ret[exp] = {'args': args, 'perplexities': perplexities}
        else:
            err += 1

    
    print("{}/ {} did not have perplexities".format(err, len(pths)))
    return ret

def load_pth_file(name):
    found_perplex = True
    perplexity = []
    try: 
        perplexity = torch.load(name)
    except Exception as e:
        found_perplex = False
    return perplexity, found_perplex
def load_json_file(name):
    found_args = True
    ret = {}
    try:
        with open(name, 'r') as f:
            ret = json.load(f)
    except Exception as e:
        print(e)
        found_args =False
    return ret, found_args

def getTest(pths, dataset='validate'):
    err = 0
    err_pths = []
    ret = {}
    for p in pths:
        exp = p.split('/')[-1]
        val_ppl, found_val_ppl = load_pth_file(p + '/{}-eval_perplexity.pth'.format(dataset))

        if 'vaenmt' in exp:
            lm_ppl, found_lm = load_pth_file(p + '/{}-lm_perplexity.pth'.format(dataset))
        else:
            lm_ppl, found_lm = [], True

        args, found_args = load_json_file(p + '/arguments.json')

        bleus, found_bleus = load_json_file(p + '/bleus-{}.json'.format(dataset))

        
        if found_val_ppl and found_args and found_lm and found_bleus:
            ret[exp] = {'args': args, 'perplexities': val_ppl, 'lm_perplexity': lm_ppl, 'bleus': bleus}
        else:
            err += 1
            err_pths.append(p)
    
    print("{}/ {} did not have {} results".format(err, len(pths), dataset))
    print('directories without {}'.format(dataset))
    for p in err_pths:
        print(p)
    return ret

if not os.path.isdir(save_dir):
    try:
        os.mkdir(save_dir)
    except FileExistsError as e:
        print("trying to make a directory that exists...moving on")


pths = getDirs(outdir)
#bleus = getBLEUS(pths)
   
results_name = outdir.split('/')[-2]
print(results_name)
#with open(save_dir + results_name + '.json', 'w') as f:
#    json.dump(bleus, f)

#perplexities = getPerplexities(pths)
#torch.save(perplexities, save_dir + results_name + '.pth')    

val_results = getTest(pths, dataset=args.dataset)
torch.save(val_results, save_dir + results_name + '-'+ args.dataset + '.pth')


# In[ ]:




