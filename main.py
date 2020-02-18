import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext import data, datasets
from utils.arguments_utils import handleInputs, write_args, load_args, get_optimization_dict
from utils.util_funcs import combineResults, filter_fn, greedy_decode, lookup_words, init_logger, generate_flow_samples
from utils.tokenizer_utils import *
from utils.custom_kl_registry import _kl_transformed_transformed

from models.ModelFactory import ModelFactory
from models.GenerativeEncoderDecoder import GenerativeEncoderDecoder
from models.VanillaJointEncoderDecoder import VanillaJointEncoderDecoder
#from util_train_funcs import train
from Trainer import Trainer, rebatch
from Translator import Translator, write_translations
from DataHandler import DataHandler
import dill
import pandas as pd
from metrics import rawBLEU, get_moses_multi_bleu
from pyro.util import set_rng_seed
import os, json, glob, copy, time, sys, argparse

import logging

print('setting global variables')
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"    
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
LOWER=False
DEVICE=torch.device('cuda') # not a big fan of this ...
USE_CUDA=True

#This is in main...because I'm superstitious and don't actually believe these things work
def setRNGSeed(seed):
    #numpy 
    np.random.seed(seed)
    #torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # sets on...curent cuda device
    torch.cuda.manual_seed_all(seed) # sets on all devices in use
    #Pyro...maybe, seems to just set them for torch again...which would make sense 
    set_rng_seed(seed)
    print("you are setting cudnn to deterministic, may make things slower")
    logging.info("cudnn is set to deterministic which may slow speed fyi")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    args = handleInputs()
    setRNGSeed(args.rng_seed)
    use_pyro = args.model_type is not 'nmt'

    if args.combine_results is not None:
        if os.path.isdir('./.results'):
            combineResults(args.combine_results)
            return
        else:
            ValueError(".results/ does not exist, assumed no experiments previously ran")
    #create directory to store experiments
    if not os.path.isdir('./.results'):
        os.mkdir('./.results')

    #create directory for dataset source to target language pair
    exp_dir = './.results/{}_{}-{}/'.format(args.dataset, args.source, args.target)
    if not os.path.isdir(exp_dir):
        try:
            os.mkdir(exp_dir)
        except FileExistsError as e:
            logging.warning("You might be trying to create {} twice (you running several runs?)".format(exp_dir))
    
    if use_pyro:
        args_name = 'kl-anneal_{}_{}_latents_{}_particles_{}_attn_{}/'.format(args.kl_anneal, args.to_anneal, args.z_dim, args.num_particles, args.use_attention)
        if args.use_flows:
            args_name = '{}_{}_'.format(args.flow_type, args.num_flows) + args_name

        exp_dir = exp_dir + '{}_'.format(args.model_type) + args_name
    else:
        exp_dir = exp_dir + 'RNNSearch/'

    #flag on whether this is an experiment continuation or not 
    if args.opt == 'test' or args.opt == 'validate':
        #if we are test or validating, it is assumed the experiment was run 1st
        args.load_latest_epoch = True
        args.load_epoch = 1

    args.load_latest_epoch = args.load_epoch >= 0 and args.load_latest_epoch
    cont_exp = args.load_epoch >= 0 or args.load_latest_epoch

    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    else:
        #there's a logic gate for this...but can't remember what it is
        if not cont_exp:
            if not args.debug:
                raise ValueError("{} already exists, if change other parameter, please rename existing file".format(exp_dir))
    #keep track of all parameters used 
    log_file = exp_dir + 'experiment.log'
    init_logger(log_file, cont_exp) 
    if cont_exp:
        logging.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logging.info("load_epoch ({}) set. Loading exp config (seems silly otherwise)".format(args.load_epoch))
        try:
            #to_pop is set to things we may want to actually update on the experiment. 
            to_pop =["load_epoch", "epochs", "print_every", "decode_alg", "k", "length_norm","load_latest_epoch", "opt","bleu_score"]
            args = load_args(args, exp_dir, to_pop=to_pop)
        except FileNotFoundError as e:
            logging.error("could not load previous arguments, are you sure you put same parameters as experiment?")
            logging.error("Starting experiment over and setting load_epoch = -1")
            args.load_epoch = -1
            args.load_latest_epoch = False
            cont_exp = False

    #whether or not we loaded arguments, presumably should also make sure things are da same
    write_args(args, exp_dir)

    optimization_dict = get_optimization_dict(args)

    
    if args.use_bpe:
        logging.info("Using BPE models : {} -> {}".format(args.src_bpe, args.trg_bpe))
        tokenize_src, tokenize_trg = getBPE(args.src_bpe, args.trg_bpe)
    else:
        logging.info("Using Tokenizer: {} -> {}".format(args.source, args.target))
        tokenize_src, tokenize_trg = getTokenizer(args.source, args.target, args.on_whitespace)

   # we include lengths to provide to the RNNs

    data_save_path = './.data/{}_data_{}_to_{}.pth'.format(args.dataset, args.source, args.target)

    datahandler = DataHandler(tokenize_src, tokenize_trg, LOWER, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN, UNK_TOKEN, args.min_freq, DEVICE)
    fields = [('src', datahandler.getSRCField()), ('trg', datahandler.getTRGField())]
    try:
        #TODO...figure out how to make this work if possible since...loading is expensive
        f = torch.load(data_save_path, pickle_module=dill)
        logging.info('found previous saved train and valid data, delete if undesired')
        datahandler.load_vocabs(f['src_vocab'], f['trg_vocab'])
        train_data= data.Dataset(f['train_examples'], fields=fields, filter_pred=None)
        valid_data = data.Dataset(f['valid_examples'], fields=fields, filter_pred=None) 
        test_data = data.Dataset(f['test_examples'], fields=fields, filter_pred=None)
    except FileNotFoundError as e:
        logging.warning('could not find previous saved file, building new one')
        if args.dataset == 'tabular':
            logging.info("Using Tabular file, assumes no header in files")
            max_len = args.max_len
            train_data, valid_data, test_data = data.TabularDataset.splits(path='./.data/',
                format='tsv',
                train='train-{}-{}.tsv'.format(args.source, args.target),
                validation='dev-{}-{}.tsv'.format(args.source, args.target),
                test='test-{}-{}.tsv'.format(args.source, args.target),
                skip_header=False,
                fields=fields, 
                filter_pred=lambda x: filter_fn(x, max_len))
        elif args.dataset == 'IWSLT':
            logging.warning("You need to create val.de-en.* and test.de-en.* by merging files before")
            train_data, valid_data, test_data = datasets.IWSLT.splits(
                exts=('.' + args.source,'.' + args.target), fields=(datahandler.getSRCField(), datahandler.getTRGField()), 
                filter_pred=lambda x: filter_fn(x, args.max_len), validation='val', test='test')
        elif args.dataset == 'WMT14':
            train_data, valid_data, test_data = datasets.WMT14.splits(
                exts=('.' + args.source,'.' + args.target), 
                fields=(datahandler.getSRCField(), datahandler.getTRGField())
            )

        datahandler.build_vocabs(train_data, args.custom_vocab_src, args.custom_vocab_trg)
        to_save = {'train_examples': train_data.examples,
                    'valid_examples': valid_data.examples,
                    'test_examples': test_data.examples,
                    'src_vocab': datahandler.getSRCVocab(),
                    'trg_vocab': datahandler.getTRGVocab()
                }
        torch.save(to_save, data_save_path, pickle_module=dill)

    logging.info('Vocab Sizes: {} (SRC) {} (TRG)'.format(len(datahandler.getSRCVocab()), len(datahandler.getTRGVocab()))) 
    logging.info('Train dataset Size: {}, Validation dataset Size: {}'.format(len(train_data), len(valid_data)))
    train_iter = datahandler.getBucketIter(train_data, batch_size=args.batch_size, train=True, 
                                    sort_within_batch=True, 
                                    sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False)

    valid_iter = datahandler.getIter(valid_data, batch_size=1, train=False, sort=False, repeat=False) 
    test_iter = datahandler.getIter(test_data, batch_size=1, train=False, sort=False, repeat=False) 

    if args.use_bpe:
        trg_bpe = spm.SentencePieceProcessor()
        trg_bpe.Load(args.trg_bpe)
        src_bpe = spm.SentencePieceProcessor()
        src_bpe.Load(args.src_bpe)
    else:
        trg_bpe = None

    if args.bleu_score == 'raw':
        bleu_func = rawBLEU
    elif args.bleu_score == 'multi':
        bleu_func = get_moses_multi_bleu


    #this is where the magic starts (hopefully)
    modelfactory = ModelFactory(len(datahandler.getSRCVocab()), len(datahandler.getTRGVocab()),
        emb_size=args.emb_size, hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, z_layer=args.z_dim, pool_size=args.max_out_dim, use_projection=args.use_projection)

    model = modelfactory.getModel(args.model_type, use_attention=args.use_attention)

    cond_flow_scale = 2
    if args.use_flows and args.model_type is not 'nmt':
        if args.flow_type == 'planar':
            model.loadPlanarFlows(args.num_flows, z_dim=args.z_dim)
        elif args.flow_type == 'iaf':
            model.loadIAFs(args.num_flows, z_dim=args.z_dim)
        elif args.flow_type == 'cond-planar':
            model.loadConditionalPlanarFlows(args.num_flows, args.hidden_size * cond_flow_scale , z_dim=args.z_dim)
        elif args.flow_type == 'cond-planar-v2':
            model.loadConditionalPlanarFlows_v2(args.num_flows, args.hidden_size * cond_flow_scale , z_dim=args.z_dim)
        elif args.flow_type == 'cond-iaf':
            model.loadConditionalIAFFlows(args.num_flows, args.hidden_size * cond_flow_scale , z_dim= args.z_dim)

    if not cont_exp:
        logging.info("Initialializing Model parameters randomly with {} scheme".format(args.init_type))
        model.initParameters(args.init_type)

    if not cont_exp:
        logging.info(model)
    if USE_CUDA:
        model = model.cuda()
    
    #some internal hacky stuff to let me do hacky things....
    model.setTrainDataSize(len(train_data))
    model.setUnkTokenIndex(datahandler.getTRGVocab().stoi[UNK_TOKEN])
    model.setSOSTokenIndex(datahandler.getSRCVocab().stoi[SOS_TOKEN])#for gnmt
    model.setPadIndex(datahandler.getSRCVocab().stoi[PAD_TOKEN])
    model.setWordDropout(args.word_dropout)
    model.setUseMeanField("Mean" in args.elbo_type)
    model.setToAnneal(args.to_anneal)
    if 'q' not in args.to_anneal and "Mean" in args.elbo_type and args.kl_anneal > 1.0:
        msg = "You are not annealing the variational distribution even though you request to anneal and are using mean field...which would use analytic form and needs to anneal q"
        logging.warning(msg)
        print(msg)

    if args.model_pth is not None:
        #model.load('./model_final.pth')
        model.load(args.model_pth)



    train_translator = Translator(valid_data, valid_iter, model, max_len=args.max_len,
        sos_index=datahandler.getTRGVocab().stoi[SOS_TOKEN], eos_index=datahandler.getTRGVocab().stoi[EOS_TOKEN],
        pad_index=datahandler.getPadIndex(), use_cuda=USE_CUDA)

    trainer = Trainer(model, train_iter, valid_iter, use_pyro, datahandler.getPadIndex(),
        train_translator, bleu_func, datahandler.getTRGVocab(), bpe_model=trg_bpe,
        use_cuda=USE_CUDA, savedir=exp_dir,
        optim_dict=optimization_dict, kl_anneal=args.kl_anneal, use_aux_loss=args.use_aux_loss,
            load_epoch=args.load_epoch, use_latest_epoch=args.load_latest_epoch)

    if args.opt == 'all' or args.opt =='train':
        dev_perplexities = trainer.train(num_epochs=args.epochs, print_every=args.print_every)
        torch.save(dev_perplexities, exp_dir + 'perplexities.pth')
    elif args.model_pth is None:
        # get best performing model
        logging.info("No model path provided, using best model for evaluation")
        dev_perplexities = trainer.initDevPerplexities()
        #if dev perplexities is not in order it was trained, this will not work
        best = {'i': -1, 'val_bleu': 0.0}
        for i, p in enumerate(dev_perplexities):
            cur_bleu = p['val_bleu']
            if cur_bleu > best['val_bleu']:
                best['i'] = i
                best['val_bleu'] = cur_bleu 
        args.model_pth = trainer.getCheckpointPth(best['i'])
        try:
            check_pt = torch.load(args.model_pth)
            model.load(check_pt['model'])
            #with mutation...this is probably not necessary, but just in case....
            trainer.setModel(model)
        except Exception as e:
            logging.warning("Failed to load a model...you do know you request to evaluate right?")
    else:
        model.load(args.model_pth)

    val_or_test = args.opt == 'all' or args.opt == 'validate' or args.opt == 'test'or args.opt =='test_lengths'

    if  val_or_test :
        if args.opt == 'test' or args.opt == 'test_lengths':
            dataset = test_data
            data_iter = test_iter
        else:
            dataset =  valid_data
            data_iter = valid_iter
        scores = {}

        debug = True 
        if val_or_test and use_pyro and debug:
            #Test utility of latent variable 
            #Another way to see how useful z is to 0 it out at translation time. That way, it gets no weight
            #This sort of test only makes sense if z is concatentaed as input at each step of decoding
            model.setUseLatent(False)
            translator = Translator(dataset, data_iter, model, max_len=args.max_len, sos_index=datahandler.getTRGVocab().stoi[SOS_TOKEN], 
                eos_index=datahandler.getTRGVocab().stoi[EOS_TOKEN], pad_index=datahandler.getPadIndex(), use_cuda=USE_CUDA, k=args.k, length_norm=args.length_norm)

            no_latent_bleu, hypotheses, references = translator.FullEvalTranslate(datahandler.getTRGVocab(), bleu_func, decodefn=args.decode_alg, bpe_model=trg_bpe)

            #store information
            no_latent_name = exp_dir + 'no-latent-{}.tsv'.format(args.opt) 
            write_translations(no_latent_name, hypotheses, references)
            scores['{}-no_latent'.format(args.opt)] = no_latent_bleu
            #subtle, but remember we need to use it after this test
            model.setUseLatent(True)

        #TODO: Probably not gonna do this...but presumably, because of mutation..., I really don't need to make another one of these...
        #Do this after the no latent test, because the Translator at this point can be used below for testing lengths
        if debug:
            translator = Translator(dataset, data_iter, model, max_len=args.max_len, sos_index=datahandler.getTRGVocab().stoi[SOS_TOKEN], 
                eos_index=datahandler.getTRGVocab().stoi[EOS_TOKEN], pad_index=datahandler.getPadIndex(), use_cuda=USE_CUDA, k=args.k, length_norm=args.length_norm)

            bleu, hypotheses, references = translator.FullEvalTranslate(datahandler.getTRGVocab(), bleu_func, decodefn=args.decode_alg, bpe_model=trg_bpe)
            logging.info("{} BLEU score: {} which was ran using {} opt".format(args.bleu_score, bleu, args.opt))
            scores[args.opt] = bleu
            translation_name = exp_dir + '{}.tsv'.format(args.opt) 
            write_translations(translation_name, hypotheses, references)

        joint_modeling = isinstance(model, GenerativeEncoderDecoder) or isinstance(model, VanillaJointEncoderDecoder)

        if joint_modeling and debug:
            model.setDecodeTarget(False)
            lm_translator = Translator(dataset, data_iter, model, max_len=args.max_len, sos_index=datahandler.getSRCVocab().stoi[SOS_TOKEN], 
                eos_index=datahandler.getSRCVocab().stoi[EOS_TOKEN], pad_index=datahandler.getPadIndex(), use_cuda=USE_CUDA, k=args.k,
                 length_norm=args.length_norm, do_lang_model=True)
            #Do greedy decoding only for language model. With these parameters, performance isn't expected to be tooo amazing 
            bleu, hypotheses, references = lm_translator.FullEvalTranslate(datahandler.getSRCVocab(), bleu_func, decodefn='greedy', bpe_model=src_bpe)
            scores["lm-{}".format(args.opt)] = bleu
            translation_name = exp_dir + 'lm-{}.tsv'.format(args.opt) 
            write_translations(translation_name, hypotheses, references)
        
        #collect validation "perplexity" for models, mostly for the ELBO 
        if joint_modeling and debug:
            def get_lm_toks():
                return trainer.model.getSRCTokCount()
            eval_perplexity = trainer.run_lvnmt_eval(trainer.rebatch_iter(data_iter), custom_tok_count=get_lm_toks, count_both=True)
            #calculate perplexity of language model 
            model.setTrainMT(False)
            model.setTrainLM(True)
            
            lm_eval_perplexity = trainer.run_lvnmt_eval(trainer.rebatch_iter(data_iter),custom_tok_count=get_lm_toks)
            torch.save(lm_eval_perplexity, exp_dir + '{}-lm_perplexity.pth'.format(args.opt))
        else:
            eval_perplexity = trainer.run_lvnmt_eval(trainer.rebatch_iter(data_iter))

        torch.save(eval_perplexity, exp_dir + '{}-eval_perplexity.pth'.format(args.opt))


        flow_samples = generate_flow_samples(trainer.model, trainer.rebatch_iter(data_iter), 
            datahandler.getSRCVocab(), datahandler.getTRGVocab(), src_bpe=src_bpe, trg_bpe=trg_bpe)
        torch.save(flow_samples, exp_dir + '{}-latent_spaces.pth'.format(args.opt))

        try:
            with open(exp_dir + 'bleus-{}.json'.format(args.opt), 'r') as bleu_scores:
                prev_bleus = json.load(bleu_scores)
        except Exception as e:
            prev_bleus = {}

        with open(exp_dir + 'bleus-{}.json'.format(args.opt), 'w') as bleu_scores:
            prev_bleus[len(prev_bleus)] = scores
            json.dump(prev_bleus, bleu_scores)

        if args.opt == 'test_lengths':
            logging.info("Calculating BLEU score based on sentence lengths")
            BLEUS = {}
            for length in range(5, 70, 5):
                references_of_length = []
                hypotheses_of_length = []
                #TODO this is stupidly inefficient... sort the ref - hypo pairs 
                for i in range(len(references)):
                    count = len(references[i].split()) 
                    if  (length-4) <= count and  count <= length:
                        references_of_length.append(references[i])
                        hypotheses_of_length.append(hypotheses[i])
                bleu = [bleu_func(hypotheses_of_length, references_of_length)]
                BLEUS['length={}'.format(length)] = bleu
            save_name = exp_dir + args.model_pth.split('/')[-1] + "_lengths.tsv"
            pd.DataFrame.from_dict(BLEUS).to_csv(save_name, sep='\t',index=False)     

    if args.opt == 'tuning':
        BLEUS = {}
        BLEUS_list = []
        for i in range(0, args.epochs):
            load_pth = exp_dir + 'checkpoints/epoch_{}.pth'.format(i)
            model.load(load_pth)
            translator = Translator(valid_data, valid_iter, model, max_len=60, sos_index=datahandler.getTRGVocab().stoi[SOS_TOKEN], 
                eos_index=datahandler.getTRGVocab().stoi[EOS_TOKEN], pad_index=datahandler.getPadIndex(), use_cuda=USE_CUDA)
            
            bleu, hypotheses, references = translator.FullEvalTranslate(datahandler.getTRGVocab(), bleu_func, decodefn='greedy',bpe_model=trg_bpe)
            BLEUS['epoch_{}'.format(i)] = [bleu]
            BLEUS_list.append(bleu)
            logging.info(load_pth)
            logging.info('{} BLEU score {}'.format(args.bleu_score,bleu))
        logging.info( "Best model for {} was {} with {} BLEU: {}".format(exp_dir, np.argmax(BLEUS_list), args.bleu_score, max(BLEUS_list)))
        save_name = exp_dir + "BLEU_scores.tsv"
        pd.DataFrame.from_dict(BLEUS).to_csv(save_name, sep='\t',index=False)



    
if __name__ == "__main__":
    main()
