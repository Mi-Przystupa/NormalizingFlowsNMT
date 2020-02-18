from Batch import Batch
from models.optim.SimpleLossCompute import SimpleLossCompute
from models.optim.SVILossCompute import SVILossCompute
import torch
import torch.nn as nn
import time, math, logging, os
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
import pyro.distributions as dist
import glob

def rebatch(pad_idx, batch, word_drop=0.0, unk_indx=-1, use_cuda=False):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.src, batch.trg, pad_idx, word_drop=word_drop, unk_indx=unk_indx, use_cuda=use_cuda)
 
class Trainer:
    def __init__(self, model, train_iter, val_iter, use_pyro, pad_index, translator, bleu_func, trg_vocab, bpe_model=None, 
     use_cuda=False, savedir='./', optim_dict={}, kl_anneal=1.0, use_aux_loss=False, load_epoch=-1, use_latest_epoch=False):

        self.model = model
        self.loss = self.createSVI(optim_dict, kl_anneal, use_aux_loss, eval=False) if use_pyro else self.createSimple(optim_dict,eval=False)
        self.dev_loss = self.createSVI(optim_dict, 1.0, use_aux_loss, eval=True) if use_pyro else self.createSimple(optim_dict,eval=True)
        self.use_pyro = use_pyro
        self.use_cuda = use_cuda
        self.savedir = savedir
        self.pad_index = pad_index
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.translator = translator
        self.trg_vocab = trg_vocab
        self.bleu_fn = bleu_func
        self.bpe_model = bpe_model
        self.last_epoch = load_epoch #if -1 then is a fresh start... 

        if load_epoch >=0 or use_latest_epoch:
            load_epoch = load_epoch if not use_latest_epoch else self.getLatestEpoch()
            self.last_epoch = self.last_epoch if not use_latest_epoch else load_epoch
            logging.info("Attempting to load epoch {}".format(load_epoch))
            load_pth = self.getCheckpointPth(load_epoch)
            try:
                self.loadCheckpoint(load_pth)
                logging.info("Successfully loaded epoch {}".format(load_epoch))
            except FileNotFoundError as e:
                logging.error("Could not find Epoch!")
                raise FileNotFoundError("Could not find epoch, set load_epoch = -1 else verify requested epoch exists in {}".format(load_pth))

    def setModel(self, model):
        self.model = model

    def getPerplexitiyPth(self):
        return self.savedir + 'perplexities.pth'
    def getLatestEpoch(self):
        pths = glob.glob(self.getCheckpointPth('*'))
        nums = [int(p.split('epoch_')[1].split('.pth')[0]) for p in pths]
        try:
            perplexities = torch.load(self.getPerplexitiyPth())
            last_epoch_with_perplex = len(perplexities) - 1
        except FileNotFoundError as e:
            #error out...means you're trying to load things that don't exist :/ 
            msg = "Perplexities not found, check experiment folder for epochs"
            logging.warning(msg)
            raise e
        #p_len -1 because load epochs start at 0
        return min(max(nums), last_epoch_with_perplex)

    def getCheckpointPth(self, epoch):
        #epoch: ideally a number which we want to increment for checkpoints
        return self.savedir + 'checkpoints/epoch_{}.pth'.format(epoch)
    
    def saveCheckpoint(self, epoch):
        name = self.getCheckpointPth(epoch)
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.loss.getOptimizerStateDict(),
            'data_loader': self.train_iter.state_dict()
        }
        if self.use_pyro:
            #SVI loss is used in this case
            step_size, kl_anneal = self.loss.getKLAnnealingSchedule()
            state['step_size'] = step_size
            state['kl_anneal'] = kl_anneal

        torch.save(state, name)

    def loadCheckpoint(self, checkpoint):
        #TODO: this may  error if you save to gpu but ask for cpu...
        state = torch.load(checkpoint)
        self.model.load_state_dict(state['model'])
        self.loss.setOptimizerStateDict(state['optimizer'])
        if self.use_pyro:
            try:
                self.loss.setKLAnnealingSchedule(step_size=state['step_size'], kl_anneal=state['kl_anneal'])
            except KeyError as e:
                step, kl_anneal = self.loss.getKLAnnealingSchedule()
                logging.error("No kl anneal saved, using initialized values: kl_anneal: {}, step: {}".format(kl_anneal,step))
        self.train_iter.load_state_dict(state['data_loader'])
        #this is because the state ends on last batch in an epoch
        _ = self.rebatch_iter(self.train_iter)

    def rebatch_iter(self, iterator, do_worddrop=False):
        worddrop = 0.0 #don't do word dropout here, it's screwing up labels #TODO remove this
        return (rebatch(self.pad_index, b, worddrop, self.model.getUnkTokenIndex(), use_cuda=self.use_cuda) for b in iterator)
   
    def createSVI(self, optim_dict, kl_anneal, use_aux_loss, eval):
        #general stuff that you always need s.t. learning rate and da elbo
        lr = optim_dict['lr']
        optimizer = optim_dict["optimizer"]
        elbo_type = optim_dict["elbo_type"]
        T = optim_dict["num_particles"]

        aux_model = None
        aux_guide = None
        if use_aux_loss:
            aux_model = self.model.aux_model
            aux_guide = self.model.aux_guide

        if optimizer == "clippedadam":
            optim_params = {'clip_norm': optim_dict['clip_norm'], "lr": lr, "betas": (optim_dict["beta1"], optim_dict["beta2"])}
        elif optimizer == "adadelta":
            optim_params = {"lr": lr, "rho": optim_dict["rho"], "eps": optim_dict["eps"]}
        elif optimizer == "clippedadadelta":
            optim_params = {"lr": lr, "rho": optim_dict["rho"], "eps": optim_dict["eps"], "clip_norm": optim_dict['clip_norm']}
        else:
            raise ValueError("{} is not supported currently", optimizer)
        loss = SVILossCompute(self.model.generator, self.model.model, self.model.guide,  optimizer,
            optim_params, elbo_type=elbo_type, num_particles=T, eval=eval, step= 1./ kl_anneal, aux_guide=aux_guide, aux_model=aux_model)
        return loss 

    def createSimple(self, optim_dict, eval):
        logging.info("NMT model just uses Adam with learning rate = 0.0003")
        print("NMT model uses Adam with lr = 0.0003")
        criterion = nn.NLLLoss(reduction="sum", ignore_index=self.pad_index)
        if not eval:
            lr = 0.0003
            optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            optim = None
        return SimpleLossCompute(self.model.generator, criterion, optim) 

    def run_epoch(self, data_iter, loss_compute, print_every=50):
        """Standard Training and Logging Function"""

        start = time.time()
        total_tokens = 0
        total_loss = 0
        print_tokens = 0

        for i, batch in enumerate(data_iter, 1):
            self.model.zero_grad()
            if not self.use_pyro:
                #the vanilla NMT model isn't done with pyro which is why we have this
                out, _, pre_output = self.model.forward(batch.src, batch.trg,
                                                    batch.src_mask, batch.trg_mask,
                                                    batch.src_lengths, batch.trg_lengths)
                loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
            else:
                loss = loss_compute(batch.src, batch.trg,
                                                    batch.src_mask, batch.trg_mask,
                                                    batch.src_lengths, batch.trg_lengths,
                                                    batch.trg_y, batch.nseqs
                                                    )  

            total_loss += loss
            total_tokens += batch.ntokens
            print_tokens += batch.ntokens
            if self.model.training and i % print_every == 0:
                elapsed = time.time() - start
                logging.info("Epoch Step: %d Loss: %f Tokens per Sec: %f Elapsed: %f" %
                        (i, loss / batch.nseqs, print_tokens / elapsed, elapsed))
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f Elapsed: %f" %
                        (i, loss / batch.nseqs, print_tokens / elapsed, elapsed))
                start = time.time()
                print_tokens = 0
        try:
            ret = math.exp(total_loss / float(total_tokens))
        except Exception as e:
            print("your loss is pretty big buddy")
            logging.warning("your loss is pretty big buddy, setting to total loss")
            ret = total_loss

        return ret
    
    def initDevPerplexities(self):
        dev_perplexities = []
        if self.last_epoch >=0:
            try:
                pth = self.getPerplexitiyPth()
                dev_perplexities = torch.load(pth)
                perp_len = len(dev_perplexities)
                epochs_done = self.last_epoch + 1 #counting from 0 can be annoying...
                #if loading an epoch that isn't last epoch, only keep epochs up till that point
                if perp_len > epochs_done:
                    logging.warning("Dev perplexity length: {}, epochs done {}".format(perp_len, epochs_done))
                    logging.warning("Keeping only perplexities from 0 to {}".format(epochs_done))
                    #should give correct # of epochs...it's annoying, but is because indexing is from 0
                    dev_perplexities = dev_perplexities[:epochs_done]
            except FileNotFoundError as e:
                dev_perplexities = [None for _ in range(self.last_epoch)]
                logging.error("Could not find previous perplexities file, packing with Nones uptill epoch") 
        return dev_perplexities

    def train(self, num_epochs=10, print_every=100):
        """Train a model"""
        if self.use_cuda:
            self.model = self.model.cuda()

        # optionally add label smoothing; see the Annotated Transformer
        dev_perplexities = self.initDevPerplexities() 

        if not os.path.isdir(self.savedir + 'checkpoints/'):
            os.mkdir(self.savedir + 'checkpoints/')
        
        #the last_epoch was technically LAST epoch ran...so name is kinda dumb...
        self.train_iter.init_epoch()
        num_epochs += (self.last_epoch + 1)
        for epoch in range(self.last_epoch+ 1, num_epochs):
            self.model.train()
            start_time = time.time()
            train_perplexity = self.run_epoch(self.rebatch_iter(self.train_iter),
                                        self.loss,
                                        print_every=print_every)
            self.saveCheckpoint(epoch)
            self.model.eval()
            with torch.no_grad():
                #BLEU score is a more useful metric...although it might slow training down a bit
                bleu, _, _ = self.translator.FullEvalTranslate(self.trg_vocab, self.bleu_fn, decodefn='greedy',bpe_model=self.bpe_model)
                batches = self.rebatch_iter(self.val_iter)
                #(rebatch(self.pad_index, b, self.model.getWordDropout(), self.model.getUnkTokenIndex()) for b in self.val_iter)
                if not self.use_pyro:
                    dev_perplexity = self.run_epoch(batches, 
                                        self.dev_loss)
                    dev_perplexity = {'dev_perplexity': dev_perplexity}
                else:
                    dev_perplexity = self.run_lvnmt_eval(batches)
                
                logging.info("Validation perplexity: %f , Validation bleu: %f" % (dev_perplexity['dev_perplexity'], bleu))
                dev_perplexity["val_bleu"] = bleu
                dev_perplexity["train_perplexity"] = train_perplexity

                dev_perplexities.append(dev_perplexity)
                torch.save(dev_perplexities, self.getPerplexitiyPth())
            msg = "Epoch: {}, duration: {}".format(epoch, time.time() - start_time)
            print(msg)
            logging.info(msg)
        return dev_perplexities

    def calc_diag_gauss_kl(self, mus_post, sigs_post, mus_prior=None, sigs_prior=None):
        #if mus_prior is None, assume it is standard multivariateNormal(0, sigs_prior)
        if mus_prior is None:
            mus_prior = torch.zeros_like(mus_post)

        #if sigs_prior is None, assume it is standard multivariateNormal(mus_prior, 1)
        if sigs_prior is None:
            sigs_prior = torch.ones_like(sigs_post)

        #below is equivalent to calling: MultivariateNormal(mus_prior,torch.diag_embed(sigs_prior.pow(2)))
        prior_gauss = dist.Normal(mus_prior,sigs_prior).to_event(1)
        pos_gaus = dist.Normal(mus_post, sigs_post).to_event(1)
        kl = kl_divergence(pos_gaus, prior_gauss)
        return kl

    def calc_analytical_gauss_kl(self, mus, sigs):
        #this is kl(q || N(0, I))
        return -0.5 * torch.sum(1 + sigs.pow(2).log() - mus.pow(2) - sigs.pow(2))

    def run_lvnmt_eval(self, data_iter, custom_tok_count=None, count_both=False):
        #this assumes you're using pyro
        start = time.time()
        total_tokens = 0
        total_loss = 0
        print_tokens = 0
        loss_terms = {'base_kl_posterior_N(0,1)': [], 'base_kl_prior_N(0,1)': [],
        'base_kl_posterior_prior': []}

        for _, batch in enumerate(data_iter, 1):
            loss = self.dev_loss(batch.src, batch.trg,
                                                batch.src_mask, batch.trg_mask,
                                                batch.src_lengths, batch.trg_lengths,
                                                batch.trg_y, batch.nseqs
                                                )  
            #this is for calculating the Kl divergence on the BASE distribution, with or without normalizing flows 
            #KL(N(mu_post, sig_post) || N(0,1))
            mu_posterior, sig_posterior = self.model.get_batch_params(ret_posterior=True)
            kl_posterior = self.calc_diag_gauss_kl(mu_posterior, sig_posterior)
            #KL(N(mu_prior, sig_prior) || N(0,1))
            mu_prior, sig_prior = self.model.get_batch_params(ret_posterior=False)
            kl_prior = self.calc_diag_gauss_kl(mu_prior, sig_prior)
            #KL(N(mu_post, sig_post) || N(mu_prior, sig_prior))
            kl_post_prior = self.calc_diag_gauss_kl(mu_posterior, sig_posterior, mu_prior, sig_prior)

            for k in loss.keys():
                if k not in loss_terms:
                    loss_terms[k] = [loss[k]] 
                else:
                    loss_terms[k].append(loss[k])
            loss_terms['base_kl_prior_N(0,1)'].append(kl_prior.item())
            loss_terms['base_kl_posterior_N(0,1)'].append(kl_posterior.item())
            loss_terms['base_kl_posterior_prior'].append(kl_post_prior.item())
        
            total_loss += loss['nll']
            if custom_tok_count is None :
                total_tokens += batch.ntokens
                print_tokens += batch.ntokens
            elif count_both:
                total_tokens += batch.ntokens + custom_tok_count()
                print_tokens += batch.ntokens + custom_tok_count()
            else:
                #TODO...hack to get token counts with GNMT
                total_tokens += custom_tok_count()
                print_tokens += custom_tok_count()
        try:
            perplexity = math.exp(total_loss / float(total_tokens))
        except Exception as e:
            print("your loss is pretty big buddy")
            perplexity = total_loss

        for k in loss_terms.keys():
            loss_terms[k] = sum(loss_terms[k])
        loss_terms['dev_perplexity'] = perplexity
        ret = loss_terms

        return ret 

