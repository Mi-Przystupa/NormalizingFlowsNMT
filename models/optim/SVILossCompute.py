import torch
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import Adagrad, PyroOptim
from pyro.optim.pytorch_optimizers import Adadelta
from models.optim.ClippedAdadelta import ClippedAdadelta
from models.optim.ClippedAdam import ClippedAdam
from pyro import poutine
import numbers
import logging

from models.optim.LossCompute import LossCompute

class SVILossCompute(LossCompute):
    """A simple loss compute and train function."""

    def __init__(self, generator, model, guide, optimizer, optim_params, elbo_type='TraceELBO', num_particles=1,
     eval=False, step = 1./ 30000.0, aux_model=None, aux_guide=None):
        optim = self.getOptimizer(optimizer, optim_params) 
        elbo = self.getELBO(elbo_type, num_particles)
        criterion = SVI(model, guide, optim, loss=elbo)
        super(SVILossCompute, self).__init__(generator, criterion, optim ) 
        
        self.eval = eval
        self.guide = guide
        self.model = model
        self.kl_anneal = step
        self.step = step
        self.aux_criterion = None
        #hack to get only KL term
        self.model_no_obs = poutine.block(model, hide=["preds", 'lm_preds'])
        optim = self.getOptimizer(optimizer, optim_params)
        elbo = self.getELBO(elbo_type, num_particles)
        self.kl_eval_svi = SVI(self.model_no_obs, self.guide, optim, elbo)
        
        #aux model and guide are for calculating additional loss terms...
        if aux_model is not None and aux_guide is not None: 
            print('setting aux loss, ')
            logging.info("setting aux loss")
            optim = self.getOptimizer(optimizer, optim_params)
            elbo = self.getELBO(elbo_type, num_particles)
            self.aux_criterion = SVI(aux_model, aux_guide, optim, loss=elbo) 

        self.aux_guide = aux_guide
        self.aux_model = aux_model
    
    def setKLAnnealingSchedule(self, step_size, kl_anneal):
        """
            step_size: how much to increase weight of KL term at each step
            beta: current weight of kl term
        """
        self.step = step_size
        self.kl_anneal = kl_anneal

    def getKLAnnealingSchedule(self):
        return self.step, self.kl_anneal
    
    def getOptimizerStateDict(self):
        return self.criterion.optim.get_state()
    
    def setOptimizerStateDict(self, state_dict):
        return self.criterion.optim.set_state(state_dict)
    
    def getELBO(self, elbo_type, particles):
        if elbo_type == 'TraceELBO':
            return Trace_ELBO(num_particles=particles)
        elif elbo_type == "MeanFieldELBO":
            return TraceMeanField_ELBO(num_particles=particles)
        else:
            raise ValueError("{} ELBO not supported".format(elbo_type)) 

    def getOptimizer(self, optimizer, optim_params):
        if optimizer == 'clippedadam':
            return PyroOptim(ClippedAdam, optim_params)
        elif optimizer == 'adadelta':
            #not 100% on this but pretty sure ** "dereferences" the dictionary 
            return Adadelta(optim_params)
        elif optimizer == 'clippedadadelta':
            #since it's custom, gotta set it up in the way Pyro expects
            return PyroOptim(ClippedAdadelta, optim_params)
        else:
            raise ValueError("{} optimizer not supported".format(optimizer))


    def __call__(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths,trg_y, norm):
        #x = self.generator(x)
        kl_anneal = self.kl_anneal
        if self.eval:
            #you could also do .eval_loss or something but this allows a bit more probing of results
            with torch.no_grad():
                elbo = self.criterion.evaluate_loss(src, trg, src_mask, trg_mask, src_lengths, trg_lengths, trg_y) * norm
                kl_term = self.kl_eval_svi.evaluate_loss(src, trg, src_mask, trg_mask, src_lengths, trg_lengths, trg_y) * norm
                nll = elbo - kl_term
                def torch_item(x):
                    return x if isinstance(x, numbers.Number) else x.item()
            
            if self.aux_criterion is not None:
                aux_loss = self.aux_criterion.evaluate_loss(src, trg, src_mask, trg_mask, src_lengths, trg_lengths, trg_y)
            else:
                aux_loss = -1.0

            loss = {'elbo': elbo, 'nll': nll, 'approx_kl': kl_term, 'aux_loss': aux_loss}

        else:
            loss = self.criterion.step(src, trg, src_mask, trg_mask, src_lengths, trg_lengths, trg_y, kl_anneal)
            if self.aux_criterion is not None:
                aux_loss = self.aux_criterion.step(src, trg, src_mask, trg_mask, src_lengths, trg_lengths, trg_y, kl_anneal)
            loss = loss * norm
            self.kl_anneal = min( self.kl_anneal + self.step, 1.0) 

        return loss 