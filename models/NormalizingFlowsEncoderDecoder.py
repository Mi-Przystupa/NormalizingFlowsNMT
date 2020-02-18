import torch
import torch.nn as nn
from models.EncoderDecoder import EncoderDecoder
from models.layers.GaussianLayer import GaussLayer
from models.layers.DenseLinear import DenseLinear
from models.layers.CondIAF import CondInverseAutoregressiveFlowStable
from models.layers.planar_v2 import ConditionalPlanarFlow_v2
import pyro
from pyro.distributions import TransformedDistribution,  Normal 
from pyro.distributions.transforms import InverseAutoregressiveFlow, PlanarFlow, ConditionalPlanarFlow
from pyro.nn import AutoRegressiveNN
from pyro.nn.dense_nn import DenseNN
from pyro import poutine
from pyro.nn import ConditionalAutoRegressiveNN

class NormalizingFlowsEncoderDecoder(EncoderDecoder):

    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(NormalizingFlowsEncoderDecoder, self).__init__(encoder, decoder, src_embed, trg_embed, generator)
        #presumably, you initially always want to use the latent variable in the beginning
        self.use_latent = True
        self.nf = []
        self.nf_modules = None
        self.surrogate_flows = []
        self.surrogate_flows_modules = None
        self.batch_size = 1.0
        self.use_cond_flow = False
        self.cached_flows = None

   
    def guide(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs, kl=1.0):
        raise ValueError("guide should be implemented by sub-class")

    def model(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths,y_trg, kl=1.0 ):
        raise ValueError("guide should be implemented by sub-class")

    def aux_guide(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs):
        raise ValueError("eval_guide should be implemented by sub-class")

    def aux_model(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs): 
        raise ValueError("eval_model should be implemented by sub-class")

    def getDistribution(self, z_mean, z_sig, cond_input, use_cached_flows = False, extra_cond = True):
        #Gets Either a multi-variate Gaussian or the transformed distribution
        #extra_cond is the extra condition in which to use the transformed distribution
        base_dist = Normal(z_mean, z_sig).to_event(1)
        if len(self.nf) > 0 and extra_cond and not self.use_cond_flow:
            dist = TransformedDistribution(base_dist, self.nf)
        elif len(self.nf ) > 0 and self.use_cond_flow and extra_cond:
            #for some reason calling flows like this has event_dim= 0 by default (is wrong)
            if (use_cached_flows and self.cached_flows is not None):
                flows = self.cached_flows
            else:
                flows = [nf.condition(cond_input) for nf in self.nf]  
                for f in flows:
                    f.event_dim = 1
                self.cached_flows = flows
            dist = TransformedDistribution(base_dist, flows)
        else:
            dist = base_dist  
        return dist

    def getVariationalDistribution(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs):
        #TODO this should be an abstract method
        return Normal(torch.zeros(10), torch.ones(10))

    def get_reconstruction_const(self, scale):
        scale = 1.0 if 'p_p' not in self.to_anneal else scale
        return self.get_normalizing_const(by_minibatch=True, by_fulldata=False, scale=scale)

    def get_guide_kl_const(self, scale):
        scale = 1.0  if not 'q' in self.to_anneal else scale
        
        return self.get_normalizing_const(by_minibatch=True, by_fulldata=False, scale=scale)
    
    def get_model_kl_const(self, scale):
        scale = 1.0 if 'p' not in self.to_anneal else scale
        return self.get_normalizing_const(by_minibatch=True, by_fulldata=False, scale=scale)


    def get_normalizing_const(self, by_minibatch, by_fulldata, scale=1.0 ):
        assert not( by_minibatch  and by_fulldata), "either normalize by minibatch OR fulldata"
        const = scale
        if by_minibatch:
            const = float(const / self.batch_size)
        elif by_fulldata:
            const = float(const / self.train_size)

        #according to a colleague the technical correct way to scale KL is 1 . / size_of_data FYI
        return const 
           
    def loadFlows(self, num_flows, flow_fn):
        self.nf =  [flow_fn() for _ in range(num_flows)]
        self.nf_modules = nn.ModuleList(self.nf)
        
    def loadIAFs(self, num_iafs, z_dim=100, iaf_dim=320):
        flow_fn =lambda : InverseAutoregressiveFlow(AutoRegressiveNN(z_dim, [iaf_dim, iaf_dim]))
        self.loadFlows(num_iafs, flow_fn)

    def loadPlanarFlows(self, num_planar, z_dim=100):
        flow_fn = lambda : PlanarFlow(z_dim)
        self.loadFlows(num_planar, flow_fn)

    def loadConditionalPlanarFlows(self, num_planar, context_dim, z_dim=100):
        flow_fn = lambda : ConditionalPlanarFlow(DenseNN(context_dim, [150], param_dims=[1, z_dim, z_dim], nonlinearity=nn.Tanh()))  
        self.use_cond_flow = True
        self.loadFlows(num_planar, flow_fn)

    def loadConditionalPlanarFlows_v2(self, num_planar, context_dim, z_dim=100):
        flow_fn = lambda : ConditionalPlanarFlow_v2(DenseNN(context_dim, [150], param_dims=[1, z_dim, z_dim], nonlinearity=nn.Tanh()))  
        self.use_cond_flow = True
        self.loadFlows(num_planar, flow_fn)

    def loadConditionalIAFFlows(self, num_iafs, context_dims, z_dim=100, iaf_dim=320):
        ar = ConditionalAutoRegressiveNN(z_dim, context_dims, [iaf_dim, iaf_dim])
        flow_fn = lambda : CondInverseAutoregressiveFlowStable(ar)
        self.use_cond_flow = True
        self.loadFlows(num_iafs, flow_fn)

    def applyFlows_(self, sample, flows, cond_inp):
        for flow in flows:
            if self.use_cond_flow:
                cond_flow = flow.condition(cond_inp)
                cond_flow.event_dim = 1 #I don't think this affects actual transformation...but just incase
                sample = cond_flow(sample)
            else:
                sample = flow(sample)

        return sample

    def applyFlows(self, sample, cond_inp):
        return self.applyFlows_(sample, self.nf, cond_inp)

    def applySurrogateFlows(self, sample, cond_inp):
        return self.applyFlows_(sample, self.surrogate_flows, cond_inp)


    def setUseLatent(self, use_latent):
        self.use_latent = use_latent