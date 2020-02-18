import argparse, json


def handleInputs():
    parser = argparse.ArgumentParser()

    #data set stuff
    parser.add_argument('--source', '-src', dest='source',
        default='de',help='2 character code for SRC language (assumed to be file extension) e.g. train.de')
    parser.add_argument('--target', '-trg', dest='target',
        default='en', help='2 character code for TRG language (assumed to be file extension e.g. train.en')
    parser.add_argument('--max_len', '-max_len', dest='max_len', default=60, type=int, 
        help="maximum sequence allowed for bitext")
    parser.add_argument('--min_freq', '-min_freq', dest='min_freq', default=1, type=int, 
        help="minimum frequency of a token necesary to be considered a word in vocabularly")
    parser.add_argument('--dataset', '-dataset', default='IWSLT', type=str,
        help='Options: IWSLT, WMT14 (de-en or en-de only), or tabular')
    parser.add_argument('--custom_vocab_src', '-custom_vocab_src', dest='custom_vocab_src', default=None, 
        help="path to a json file which contains a dictionary , where key=word, value=index,  e.g. {dog: 0}, make sure min_freq=1")
    parser.add_argument('--custom_vocab_trg', '-custom_vocab_trg', dest='custom_vocab_trg', default=None, 
        help="path to a json file which contains a dictionary , where key=word, value=index,  e.g. {dog: 0}, make sure min_freq=1")
    parser.add_argument('--on_whitespace', '-on_whitespace', dest='on_whitespace', default=False, action="store_true",
        help="flag to use whitespaces to tokenize sentences") 

    #experiment stuff 
    parser.add_argument('--operation', '-opt', dest='opt', 
        default='all', help='options: all (does train & validate) ,train, validate, test, tuning, test_lengths')
    parser.add_argument('--model_pth', dest='model_pth',
        default=None, help='name of model file to load')
    parser.add_argument('--bleu_score', '-bleu_score', dest='bleu_score', default='raw',
        help= "options are raw or  multi")
    parser.add_argument('--rng_seed', '-rng_seed', '--seed', '-seed', dest='rng_seed', default=42, type=int,
        help="random number generator seed, supposedly helps with reproducing results sets for pytorch, numpy and... Pyro?? (not totally confident for pyro)")
    parser.add_argument('--combine_results', '-combine_results', default=None, 
        help='name of dataset results to combine (e.g. dataset_IWSLT_de_en). .results/ must exist prior to calling')
    parser.add_argument('--load_epoch', '-load_epoch', dest='load_epoch', default=-1, type=int,
        help='If you want to continue training a model, you need to specify this value, assumes experiment already exists and will overwrite most args. Assumes yous specified params to match exp folder name')
    parser.add_argument('--load_latest_epoch', dest='load_latest_epoch', action="store_true", 
        default=False, help="flag to load latest epoch in an experiment run, needs load_epoch >=0 but it's value is otherwise ignored")
    
    #setting BPE
    parser.add_argument('--use_bpe', '-bpe', dest='use_bpe', action="store_true",
        default=False,help='Flag to use byte-pair encodings, requires src_bpe and trg_bpe to be set')
    parser.add_argument('--src_bpe', '-src_bpe', dest='src_bpe',
        default=None, help='path to src_bpe model, model must have been created with SentencePiece, ignored if use_bpe not set')
    parser.add_argument('--trg_bpe', '-trg_bpe', dest='trg_bpe', 
        default=None, help='path to trg_bpe model, model must have been created with SentencePiece, ignored if use_bpe not set')

    #General Model config 
    parser.add_argument('--model_type', '-model_type', dest='model_type', default='vnmt',
        help='Current Options are nmt, vanilla_nmt, vnmt, mod_vnmt, simple_mod_vnmt, gnmt, vaenmt, mod_vaenmt ( more to maybe come )' )
    parser.add_argument('--hidden_size', '-hidden_size', dest="hidden_size", default=64, type=int,
        help="Size of hidden layers, this is ...like for every aspect of the model's hidden layers" )
    parser.add_argument('--num_layers', '-num_layers', dest="num_layers", default=1, type=int,
        help="number of layers in encoder and decoder")
    parser.add_argument('--dropout', '-dropout', default=0.0, dest="dropout", type=float,
        help="amount of dropout through the model (same value everywhere dropout used")
    parser.add_argument("--emb_size", "-emb_size", default=620, dest="emb_size", type=int,
        help="word embedding dimensions")
    parser.add_argument('--init_type', '-init_type', default='normal', dest='init_type',
        help="Initialization type for all model parameters, options are: normal, xavier_uniform. Note that rnns are initialized as orthogonal matrices regardless of choice")
    parser.add_argument('--max_out_dim', '-max_out_dim', default=2, type=int, dest='max_out_dim',
        help='the step size max_out will do over the penultimate layer to the projection layer to vocabularly...should probably leave as 2')
    parser.add_argument('--use_attention', default=True, type=lambda x: bool(int(x)), dest="use_attention", 
        help="whether to use attention or not, 0=False, 1=True")
    
    #Normalizing flows
    parser.add_argument('--use_projection', '-use_projection', dest='use_projection', default=False,
        action="store_true", help='flag to add affine transform between samples from latent variable to components in VNMT/GNMT models' )
    parser.add_argument('--z_dim', '-z_dim', dest='z_dim', default=128, type=int,
        help='Dimensions of the latent space used for experiments')
    parser.add_argument('--use_flows', '-uf', dest='use_flows', action="store_true",
        default=False, help='flag to use normalizing flows or not')
    parser.add_argument('--num_flows', '-nf', dest='num_flows', 
        default=2, type=int, help='Number of flows to add to latent model (model must be vnmt or gnmt)')
    parser.add_argument('--flow_type', '-nft', dest='flow_type',
        default='planar', help='type of normalizing flow to use. curr options: planar,iaf,cond-planar')
    parser.add_argument('--to_anneal', "-to_anneal", dest="to_anneal",  default='q_p',
        help="options: q_p, q, or p , most logical choices are q_p or p. affects whether variational q or p will be annealed if annealing used")

    #Latent Variable stuff...mostly with loss
    parser.add_argument('--use_aux_loss', '-use_aux_loss', '--aux_loss', '-aux_loss', dest="use_aux_loss",
     action="store_true", default=False, help="flag to use auxilliary loss (you have to define aux_model & aux_guide on model")

    #Parameters for dealing with mode collapse problem 
    parser.add_argument('--word_dropout', '-wd', '-word_dropout', dest='word_dropout', default=0.0,
        type=float, help="number between 0.0 - 1.0 which is the probability words are replaced with the <unk> token")
    parser.add_argument('--kl_anneal', '-kl_anneal', dest='kl_anneal', default= 1.0, 
        type=float, help="val range: 1.0 <= kl_anneal, number of mini batch updates to do kl_annealing, follows linear schedule (curr + 1 / (# kl anneal steps)")
    
    #Optimization Parameters
    parser.add_argument('--epochs', '-e', '-epochs', dest='epochs', default=15, type=int,
        help="Number of epochs to train model for")
    parser.add_argument('--batch_size', '-batch_size', '-batch', dest='batch_size', default=128, type=int,
        help="size of the batch to use, should probably be at least 100 to ignore number of samples problem...")
    parser.add_argument('--print_every', '-print_every', dest='print_every', default=100, type=int,
        help="Integer to decide how often to print training perplexity")
    
    #Optimizer Parameters
    parser.add_argument('--optimizer', '-optimizer', dest='optimizer', default='adadelta', type=str,
       help="Optimiezer choices: clippedadam, adadelta, clippedadadelt")
    parser.add_argument('--rho','--rho',dest="rho", default=0.95, type=float,
        help="rho value for adadelta optimizers" )
    parser.add_argument('--eps', '-eps', default=1e-6, type=float, dest='eps', 
        help="epsilon for adadelta algorithm")
    parser.add_argument('--beta1', '-beta1', default=0.9, type=float, dest="beta1",
        help="one of the betas in cliped adam")
    parser.add_argument('--beta2', '-beta2', default=0.999, type=float, dest="beta2",
        help="one of the betas in cliped adam")
    parser.add_argument('-learning_rate', '--learning_rate', '-lr', '--lr', dest='lr', type=float, default=1.0,
        help="learning rate, used for both optimizers" )
    parser.add_argument('--clip_norm', '-clip_norm', default=20.0, type=float, dest='clip_norm',
        help="the clipping range for gradients for optimizers with Clipped in their name")
    
    #Decoding Parameters
    parser.add_argument('--decode_alg', '-decode_alg', default='greedy', dest='decode_alg', type=str, 
        help="Algorithm to use for decodeing choices: greedy, beamsearch")
    parser.add_argument('--k', '-k', default=10, type=int, dest='k', 
        help="number of active beams to consider when using beamsearch")
    parser.add_argument('--length_norm', '-length_norm', default=False, dest='length_norm', action='store_true', 
        help='Whether to consider normalizing beamsearch scores by length') 

    #ELBO parameters
    parser.add_argument('--num_particles', '-num_particles', dest='num_particles', default=1, type=int,
        help="Number of particles to use to approximate elbo (not used right now)")
    parser.add_argument('--elbo_type', '-elbo_type', dest='elbo_type', default='TraceELBO', type=str,
        help='Choices: TraceELBO, MeanFieldELBO'  )

    #debug
    parser.add_argument('--debug', action="store_true", default=False, help="keeps code from erroring on file creation stuff everytime it's rerun")

    return parser.parse_args()

def write_args(args, dir):
    arg_dict = vars(args)
    with open(dir + 'arguments.json', 'w') as f:
        json.dump(arg_dict, f, indent=4)

def load_args(args, dir, to_pop=[]):
    #pretty sure objects are mutable so the return is excessive...
    #but I'm also paranoid and don't believe in python
    with open(dir + 'arguments.json', 'r') as f:
        exp_args = json.load(f)
    
    for t_p in to_pop:
        exp_args.pop(t_p, None)

    for k,v in exp_args.items():
        setattr(args, k, v)
    return args

def get_optimization_dict(args):
    #given an argument dictionary, return assumed optimization parameters
    #note that this WILL error if any params in list are no in dict
    #also tracks elbo specific stuff...fyi
    arg_dict = vars(args)
    optimizer_params = ["optimizer", "rho", "eps", "beta1", "beta2", "lr", "clip_norm", "elbo_type", "num_particles"]
    return {k: arg_dict[k] for k in optimizer_params}
    