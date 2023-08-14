import warnings
import os
from model_function import *
from utils import *
from torch_geometric.loader import DataLoader
import torch.nn.functional as Fin
import timeit
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib 
from torch_geometric.data import Data
from torchdiffeq import odeint as odeint
import matplotlib
matplotlib.use('Agg')
import argparse
import os
import time
import torch
import torch.optim as optim

SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams',"scipy_solver","adaptive_heun"]
parser = argparse.ArgumentParser('PDE based Antibody sequence-structure generation')

parser.add_argument('--solver', type=str, default="adaptive_heun", choices=SOLVERS)
parser.add_argument('--atol', type=float, default=5e-1)
parser.add_argument('--rtol', type=float, default=5e-1)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--mask', type=int, default=0,help="How much % of Antigen labels to mask")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--cdr', type=int, default=1)
args = parser.parse_args()

cwd = os.getcwd() 
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')



test_path = str(cwd) + "/data/sabdab_new/cdrh"+ str(args.cdr) + "/test.json"

model = torch.load(str(cwd) + "/checkpoints/Abode_cond_cdrh"+str(args.cdr)+".pt") # loading the saved model
t_begin=0.
t_end=1
t_nsamples=200
t_space = np.linspace(t_begin, t_end, t_nsamples) # time-steps for the ODE function
print("################# Region given is CDR",args.cdr," ###############")
print("############################ Data is loading ###########################")
Test_data = get_graph_data_polar_with_sidechains_angle(args.cdr,test_path,args.mask) # loading the dataset 
Test_loader = DataLoader(Test_data, batch_size=1, shuffle=False,num_workers=0)

model.eval()
ppl_pred = []
rmsd_pred = []
RMSD_test_n = []
RMSD_test_ca = []
RMSD_test_ca_cart = []
RMSD_test_c = []
Perplexity = []
PPL_final = []
RMSD_final = []

for idx,batch in enumerate(Test_loader):
            data = batch.x.to(device)
            t = torch.tensor(t_space).to(device)
            params_list = [batch.edge_index.to(device),batch.order.to(device),batch.a_index.to(device)]
            model.update_param(params_list)
            options = {
                'dtype': torch.float64,
                # 'first_step': 1.0e-9,
                # 'grid_points': t,
            }

            adjoint_options = {
                'norm': "seminorm"
            }

            y_pd = odeint(
               model, data, t, method=args.solver, 
                rtol=args.rtol, atol=args.atol,
                options=options
            ) # The ODE-function to solve the ODE-system
            
            y_gt = batch.y.to(device)
            antigen_len = []
            for entry in batch.ag_len.numpy().tolist():
                antigen_len.append(entry[0])

            antibody_len = []
            for entry in batch.ab_len.numpy().tolist():
                antibody_len.append(entry[0])
                
            final_pred = get_antibody_entries(y_pd[-1],batch.batch,antibody_len,antigen_len) # get antibody entries from the total graph
            rmsd_n,rmsd_ca,rmsd_c,acc,rmsd_cart_ca = evaluate_rmsd_with_sidechains_cond_angle(final_pred,y_gt,batch.first_res) # metric function to calculate the metrics
            ppl_pred.append(acc)
            rmsd_pred.append(rmsd_ca)
            RMSD_test_n.append(rmsd_n)
            RMSD_test_ca.append(rmsd_ca)
            RMSD_test_ca_cart.append(rmsd_cart_ca)
            RMSD_test_c.append(rmsd_c)
            Perplexity.append(acc)

RMSD_test_arr_n = np.array(RMSD_test_n).reshape(-1,1) 
RMSD_test_arr_ca = np.array(RMSD_test_ca).reshape(-1,1)
RMSD_test_arr_ca_cart = np.array(RMSD_test_ca_cart).reshape(-1,1)
RMSD_test_arr_c = np.array(RMSD_test_c).reshape(-1,1)
Perplexity_arr = np.array(Perplexity).reshape(-1,1)

print("Max Recovery ",max(Perplexity).item(), " | Mean Recovery ", np.mean(Perplexity_arr,axis=0)[0], " | Std Recovery", np.std(Perplexity_arr,axis=0)[0])
print("Min RMSD ",min(RMSD_test_ca_cart), " | Mean RMSD ", np.mean(RMSD_test_arr_ca_cart,axis=0)[0], " | Std RMSD ", np.std(RMSD_test_arr_ca_cart,axis=0)[0])
