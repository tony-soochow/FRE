import wandb
import torch
from torch.utils.data import TensorDataset
import numpy as np
import logging
import base_model
import random
import argparse
import tools
import torch.nn as nn
import torch.nn.functional as F

import os
import tabNet

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import f_classif

from sklearn.decomposition import PCA
# from transformer import EncoderLayer
from tab_transformer import TabTransformer
# from fair_clustering import my_kmeans

logging.basicConfig(filename='compas_smooth_2.log', level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s=>%(message)s')
logger = logging.getLogger(__name__)


def z_sampling_(mean, log_var, device):
        eps = torch.randn(mean.size()[0], mean.size()[1], device=device)
        return eps * torch.exp(log_var / 2) + mean

class FairRep(torch.nn.Module):

    def __init__(self, input_size, a_dim, config):
        super(FairRep, self).__init__()
        z_dim = config.zdim
        hidden = (32, 32)
        self.t = 0.1
        self.a_dim = a_dim
        # import pdb;pdb.set_trace()
        self.encoder =torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),          
            torch.nn.BatchNorm1d(128),                 
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),                    
            torch.nn.BatchNorm1d(128),                
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),                    
            torch.nn.Linear(128, 40)                 
        )

        self.cla = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden[0]),
            torch.nn.Linear(hidden[0], 2),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(20, 16),
            torch.nn.Linear(16, input_size)
        )

        self.new_net = torch.nn.Sequential(
            torch.nn.Linear(input_size+1, input_size),
            torch.nn.ReLU(),
        )
        self.recon_lossf = nn.BCELoss() 

    def distance(self, x, y):
        x_mu = torch.mean(x, dim=0)
        y_mu = torch.mean(y, dim=0)
        x_var = torch.var(x, dim=0)
        y_var = torch.var(y, dim=0)
        static_ = torch.sum((x_mu - y_mu) ** 2) + torch.sum((x_var - y_var) ** 2)
        return static_

    # Loss_RG
    def twin_loss_fun(self, z, y, s):
        twin_loss = 0
        for yyy in range(2):
            ids = torch.where(y == yyy)[0]
            z_y = z[ids]
            z_a = s[ids]
            a_tag = torch.argmax(z_a, dim=1)
            mus = []
            for j in range(s.shape[1]):
                iids = torch.where(a_tag == j)[0]
                points_j = z_y[iids]
                mus.append(points_j)

            for ii in range(len(mus) - 1):
                g1 = mus[ii]
                g2 = mus[ii + 1]
                if g1.shape[0] < 2 or g2.shape[0] < 2:
                    continue
                twin_loss += self.distance(g1, g2)
        return twin_loss

    # Loss_AG
    def loss_t(self, z, y, s):
        twin_loss = 0
        for yyy in range(2):
            ids = torch.where(y == yyy)[0]
            z_y = z[ids]
            z_a = s[ids]
            a_tag = torch.argmax(z_a, dim=1)
            mus = []
            for j in range(s.shape[1]):
                iids = torch.where(a_tag == j)[0]
                points_j = z_y[iids]
                mus.append(points_j)

            for ii in range(len(mus) - 1):
                g1 = mus[ii]
                g2 = mus[ii + 1]
                if g1.shape[0] < 2 or g2.shape[0] < 2:
                    continue
                for k in range(g1.shape[1]):
                    twin_loss += self.corr(g1[:, k], g2[:, k])
        return twin_loss

    # Loss_CE
    def cla_diff(self, z, y, s):
        cla_diff_loss = 0
        a_tag = torch.argmax(s, dim=1)
        mus = []
        for j in range(s.shape[1]):
            iids = torch.where(a_tag == j)[0]
            z_j = z[iids]
            y_j = y[iids]
            mus.append(torch.nn.CrossEntropyLoss(reduction='mean')(self.cla(z_j), y_j.long()))

        for ii in range(len(mus) - 1):
            g1 = mus[ii]
            g2 = mus[ii + 1]
            cla_diff_loss += abs(g1 - g2)

        return cla_diff_loss

    def corr(self, x, y):
        """
        相关系数 越低，越不相关
        """
        xm, ym = torch.mean(x), torch.mean(y)
        if  xm == 0: return 0
        xvar = torch.sum((x - xm) ** 2) / (x.shape[0] - 1)
        # xvar = torch.sum((x - xm) ** 2) / x.shape[0]
        yvar = torch.sum((y - ym) ** 2) / (x.shape[0] - 1)
        # import pdb; pdb.set_trace()
        return torch.abs(torch.sum((x - xm) * (y - ym)) / (xvar * yvar) ** 0.5)

    def kld_exact(self, mu_p, ss_p, mu_q, ss_q):
        '''kld computing function'''
        p = MultivariateNormal(mu_p, torch.diag_embed(ss_p))
        q = MultivariateNormal(mu_q, torch.diag_embed(ss_q))
        return (kl_divergence(p, q)/mu_p.shape[1]).mean()

    def dcd_g(self, d_zx, d_zs, d_xs, cont_xs=1):
        '''G kernel Distributional Contrastive Disentangle loss'''
        if cont_xs:
            # import pdb; pdb.set_trace()
            dxs1, dxs2 = d_xs
            return d_zx + torch.exp(-d_zs) + torch.exp(-dxs1) + torch.exp(-dxs2)
        else:
            return d_zx + torch.exp(-d_zs)

    def compute_bi_kld(self, mu_x1, logvar_x1, mu_s1, logvar_s1, mu_x2, logvar_x2, mu_s2, logvar_s2):
        '''averaged kld (symmetrize)'''
        ss_x1, ss_s1, ss_x2, ss_s2 = logvar_x1.exp(), logvar_s1.exp(), logvar_x2.exp(), logvar_s2.exp()

        # Div(Zx, Zx') - must be sim
        d_zx_l = self.kld_exact(mu_x1, ss_x1, mu_x2, ss_x2)
        d_zx_r = self.kld_exact(mu_x2, ss_x2, mu_x1, ss_x1)
        d_zx = (d_zx_l + d_zx_r) / 2

        # Div(Zs, Zs') - must be diff
        d_zs_l = self.kld_exact(mu_s1, ss_s1, mu_s2, ss_s2)
        d_zs_r = self.kld_exact(mu_s2, ss_s2, mu_s1, ss_s1)
        d_zs = (d_zs_l + d_zs_r) / 2

        if mu_x1.shape == mu_s1.shape:  # Zx <-> Zs 
            # Div(Zx, Zs) - must be diff
            d_xs_ori_l = self.kld_exact(mu_x1, ss_x1, mu_s1, ss_s1)
            d_xs_ori_r = self.kld_exact(mu_s1, ss_s1, mu_x1, ss_x1)
            d_xs_ori = (d_xs_ori_l + d_xs_ori_r) / 2

            # Div(Zx', Zs') - must be diff
            d_xs_cont_l = self.kld_exact(mu_x2, ss_x2, mu_s2, ss_s2)
            d_xs_cont_r = self.kld_exact(mu_s2, ss_s2, mu_x2, ss_x2)
            d_xs_cont = (d_xs_cont_l + d_xs_cont_r) / 2

            d_xs = d_xs_ori, d_xs_cont
            return d_zx, d_zs, d_xs
        else:
            return d_zx, d_zs, torch.tensor(0.0)

    def distribution_to_var(self, model_input_ori, model_input_cons, is_add_y_pred = 0):
        if is_add_y_pred:
            zt1 = self.new_net(model_input_ori)
            zt2 = self.new_net(model_input_cons)
        else:
            zt1 = self.encoder(model_input_ori)
            zt2 = self.encoder(model_input_cons)
        out11 = zt1[:, :20]
        out12 = zt1[:, 20:]
        out21 = zt2[:, :20]
        out22 = zt2[:, 20:]
        (mu_x1, mu_s1, logvar_x1, logvar_s1) = out11[:, :10], out11[:, 10:], \
                                           out12[:, :10], out12[:, 10:]
        (mu_x2, mu_s2, logvar_x2, logvar_s2) = out21[:, :10], out21[:, 10:], \
                                           out22[:, :10], out22[:, 10:]
        
        return mu_x1, mu_s1, logvar_x1, logvar_s1, mu_x2, mu_s2, logvar_x2, logvar_s2

    def forward(self, sample):
        x, y, a = sample
        y = y.squeeze()

        model_input_ori = torch.cat((x, a), dim=1)
        model_input_cons = torch.cat((x, 1-a), dim=1)
        mu_x1, mu_s1, logvar_x1, logvar_s1, mu_x2, mu_s2, logvar_x2, logvar_s2 = self.distribution_to_var(model_input_ori, model_input_cons)

        d_zx, d_zs, d_xs = self.compute_bi_kld(mu_x1, logvar_x1, mu_s1, logvar_s1, mu_x2, logvar_x2, mu_s2, logvar_s2)

        zx1 = z_sampling_(mu_x1, logvar_x1, 'cpu')
        zs1 = z_sampling_(mu_s1, logvar_s1, 'cpu')
        zx2 = z_sampling_(mu_x2, logvar_x2, 'cpu')
        zs2 = z_sampling_(mu_s2, logvar_s2, 'cpu')

        # reconstruct x s
        recon1 = self.decoder(torch.cat((zx1, zs1), dim=1))
        recon1 = torch.sigmoid(recon1)

        recon_x1 = recon1[:, :-self.a_dim]
        recon_s1 = recon1[:, -self.a_dim:].reshape(-1, self.a_dim)

        y_pred = torch.argmax((self.cla(zx1)+self.cla(zx2))*0.5, dim=1)
        # import pdb; pdb.set_trace()
        model_input_ori2 = torch.cat((x, a, y_pred.unsqueeze(1)), dim=1)
        model_input_cons2 = torch.cat((x, 1-a, y_pred.unsqueeze(1)), dim=1)

        output_ori = self.new_net(model_input_ori2)
        output_cons = self.new_net(model_input_cons2)

       
        mu_x1, mu_s1, logvar_x1, logvar_s1, mu_x2, mu_s2, logvar_x2, logvar_s2 = self.distribution_to_var(output_ori, output_cons)
        d_zx, d_zs, d_xs = self.compute_bi_kld(mu_x1, logvar_x1, mu_s1, logvar_s1, mu_x2, logvar_x2, mu_s2, logvar_s2)

        
        zx1 = z_sampling_(mu_x1, logvar_x1, 'cpu')
        zs1 = z_sampling_(mu_s1, logvar_s1, 'cpu')
        zx2 = z_sampling_(mu_x2, logvar_x2, 'cpu')
        zs2 = z_sampling_(mu_s2, logvar_s2, 'cpu')


        # reconstruct x s
        recon1 = self.decoder(torch.cat((zx1, zs1), dim=1))
        recon2 = self.decoder(torch.cat((zx2, zs2), dim=1))
        recon1 = torch.sigmoid(recon1)
        recon2 = torch.sigmoid(recon2)

        recon_x1 = recon1[:, :-self.a_dim]
        recon_s1 = recon1[:, -self.a_dim:].reshape(-1, self.a_dim)
        recon_x2 = recon2[:, :-self.a_dim]
        recon_s2 = recon2[:, -self.a_dim:].reshape(-1, self.a_dim)

        x_ori = torch.sigmoid(x)
        a_ori = torch.sigmoid(a)

        recon_x_loss = 0.5*(self.recon_lossf(recon_x1, x_ori) + self.recon_lossf(recon_x2, x_ori))
        recon_s_loss = 0.5*(self.recon_lossf(recon_s1, a_ori) + self.recon_lossf(recon_s2, a_ori))
        recon_loss = recon_x_loss + recon_s_loss

        # swap reconstruct
        recon1_swap = self.decoder(torch.cat((zx1, zs2), dim=1))
        recon2_swap = self.decoder(torch.cat((zx2, zs1), dim=1))
        recon1_swap = torch.sigmoid(recon1_swap)
        recon2_swap = torch.sigmoid(recon2_swap)

        ms_ori = self.recon_lossf(recon1_swap, torch.cat((x_ori, a_ori), dim=1))
        ms_cont = self.recon_lossf(recon2_swap, torch.cat((x_ori, torch.sigmoid(1-a)), dim=1))
        sr_loss = (ms_ori + ms_cont) / 2
        dis_loss = self.dcd_g(d_zx, d_zs, d_xs)
        twin_loss = self.twin_loss_fun(zx1, y, a)
        a_tags = torch.argmax(a, dim=1).float()
        class_loss = (torch.nn.CrossEntropyLoss(reduction='none')(self.cla(zx1), y.long()) + torch.nn.CrossEntropyLoss(reduction='none')(self.cla(zx2), y.long())) / 2
        
        return torch.sum(class_loss), config.f1 * self.corr(a_tags, class_loss), 0.1 * twin_loss, dis_loss.mean(), recon_loss, sr_loss
       

def train(m, opt, epochs):
    for i in range(epochs):
        sr = tools.LossRecoder()
        
        for sample in ds.train_loader:
           
            l1, l2, l3, l4, l5, l6= m(sample)
            
            sr.add(l1, l2, l3, l4, l5, l6)
            loss = l1 + l2 + l3 + l4 + l5 + l6
            
            opt.zero_grad()
            loss.backward()
            opt.step()

        if i % 10 == 0:
            res = test(m, *ds.test_loader.dataset.tensors)
            print(res)
            logging.info(f'{res}')

def test(m, x, y, s):
    with torch.no_grad():

        model_input = torch.cat((x, s), dim=1)

        zt = m.encoder(model_input)
        out1 = zt[:, :20]
        out2 = zt[:, 20:]
        (mu_x, mu_s, logvar_x, logvar_s) = out1[:, :10], out1[:, 10:], \
                                           out2[:, :10], out2[:, 10:]

        zx = z_sampling_(mu_x, logvar_x, 'cpu')
        zs = z_sampling_(mu_s, logvar_s, 'cpu')

        # reconstruct x s
        recon1 = m.decoder(torch.cat((zx, zs), dim=1))
        recon1 = torch.sigmoid(recon1)

        recon_x1 = recon1[:, :-m.a_dim]
        recon_s1 = recon1[:, -m.a_dim:].reshape(-1, m.a_dim)

        y_pred = torch.argmax(m.cla(zx), dim=1)
        input_ori = torch.cat((x, recon_s1, y_pred.unsqueeze(1)), dim=1)

        model_input2 = m.new_net(input_ori)

       
        zt2 = m.encoder(model_input2)
        out1 = zt2[:, :20]
        out2 = zt2[:, 20:]
        (mu_x, mu_s, logvar_x, logvar_s) = out1[:, :10], out1[:, 10:], \
                                           out2[:, :10], out2[:, 10:]
        
        # import pdb; pdb.set_trace()
        zx = z_sampling_(mu_x, logvar_x, 'cpu')
        zs = z_sampling_(mu_s, logvar_s, 'cpu')

        recon1 = m.decoder(torch.cat((zx, zs), dim=1))
        recon1 = torch.sigmoid(recon1)

        recon_x1 = recon1[:, :-m.a_dim]
        recon_s1 = recon1[:, -m.a_dim:].reshape(-1, m.a_dim)

        return mtc(model_output=torch.softmax(m.cla(zx), dim=1).numpy()[:, 1], samples=(
            x.numpy(), y.squeeze().numpy(), s.numpy()
        ))


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

class AF:
    def __init__(self):
        pass


if __name__ == '__main__':
    os.environ["WANDB_API_KEY"] = "34317113456ff8945f86d4a9e46e64605c38198b"
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="twin-fair", entity
               ="test_bfa")

    config = wandb.config
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--data', type=str, default='compas')
    parser.add_argument('--f1', type=float, default=1)
    parser.add_argument('--f2', type=float, default=1)

    args = parser.parse_args()
    seed_everything(args.seed)
    print(args)

    "************* setting configs *************"
    config.batch_size = 64  # fixed
    config.method = 'BFA'  # fixed
    config.zdim = 10  # 10
    config.data = args.data
    config.epoch = args.epoch
    config.seed = args.seed
    config.f1 = args.f1
    config.f2 = args.f2

    "************* loading data *************"
    ds = tools.DataStream(config.data)
    print(ds.train_loader.dataset.tensors[2].mean(dim=0))
    atags = torch.argmax(ds.train_loader.dataset.tensors[2], dim=1)
    print(ds.train_loader.dataset.tensors[1][atags == 0].mean())
    print(ds.train_loader.dataset.tensors[1][atags == 1].mean())
    # import pdb; pdb.set_trace()
    "************* train and test *************"
    model = FairRep(ds.x_dim+ds.a_dim, ds.a_dim, config)
    opt = torch.optim.Adam(model.parameters())
    mtc = base_model.Metrics('acc', 'dp', 'dp2', 'ap', 'ap2', 'di', 'eo', 'eo2')
    train(model, opt, config.epoch)
    res = test(model, *ds.test_loader.dataset.tensors)

    wandb.log(res)
    print('final TEST:', res)
    logging.info(f'{res}')
    wandb.watch(model)
