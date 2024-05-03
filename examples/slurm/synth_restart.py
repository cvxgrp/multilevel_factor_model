import numpy as np
import random

import timeit, warnings, argparse, os, glob

import mfmodel as mfm
import numba as nb
import pickle



np.random.seed(1001)
random.seed(1001)


# Ignore specific warning
warnings.filterwarnings("ignore", message="omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.")

# Ignore NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=nb.NumbaPerformanceWarning)


parser = argparse.ArgumentParser(description='Solve a problem based on rank and n.')
parser.add_argument('--slurm', type=int, default=0)
args = parser.parse_args()

n = [5*10**3, 10**4, 5*10**4, 10**5][args.slurm - 1]


mtype = "large_smfm_n%d"%n



signal_to_noise = 4


L = 6


ranks = np.array([30, 20, 10, 5, 2, 1])
rank = ranks.sum()

nsamples = rank * 5
nsamples


file_name = os.path.join("/home/groups/boyd/tetianap/mfm", 'outputs/%s_rank%d_L%d.pickle' % (mtype, rank, L))
with open(file_name, 'rb') as handle:
    d1 = pickle.load(handle)
    F0 = d1["F"]
    D0 = d1["D"]
    true_F = d1["true_F"]
    true_D = d1["true_D"]
    loglikelihoods = d1["ll"]
    true_ll = d1["true_ll"]
    F_hpart = d1["hpart"]

mfm.valid_F_hpart(F_hpart)


true_mfm = mfm.MFModel(hpart=F_hpart, F=true_F, D=true_D)
F_hpart["pi_inv"] = true_mfm.pi_inv

print(f"{n=}, {true_mfm.num_factors()=}, {L=}, {ranks.sum()=}")


# EM 

C = mfm.sample_data(nsamples, true_mfm)
Z = (C - C.mean(axis=1, keepdims=True))[F_hpart["pi"], :]

# permute to put clusters on diagonal
Y = Z.T
N = Y.shape[0]

permuted_F_hpart = {"pi_inv":np.arange(n), "pi":np.arange(n), "lk":F_hpart["lk"]}
row_selectors, si_groups, F_hpart, groups_all = mfm.row_col_selections(hpart, return_groups=True)


true_train_obj = mfm.fast_loglikelihood_value(true_mfm.F, true_mfm.D, Y, ranks, permuted_F_hpart, true_mfm.num_factors(),
                                           tol1=1e-5, tol2=1e-5)
exp_true_ll = mfm.fast_exp_true_loglikelihood_value(true_mfm.F, true_mfm.D, ranks, permuted_F_hpart, true_mfm.num_factors(),
                                           tol1=1e-12, tol2=1e-12)
print(f"True: {true_ll}")
true_ll = {"train":true_train_obj, "exp":exp_true_ll}
print(f"True: {true_ll}")


mfm_Sigma = mfm.MFModel(hpart=F_hpart, ranks=ranks, F=F0, D=D0)


# script_dir = os.getcwd()
script_dir = "/home/groups/boyd/tetianap/mfm"
# parent_dir = os.path.dirname(script_dir)
output_path = os.path.join(script_dir, 'outputs/%s_rank%d_L%d_s%d.pickle' % (mtype, rank, L, mfm_Sigma.num_factors()))

N = Y.shape[0]
eps = 1e-12
for t in range(100):
    F1 = mfm.fast_EM_get_F(F0, D0, Y, ranks, permuted_F_hpart, row_selectors, si_groups)
    D1 = mfm.fast_EM_get_D(F0, D0, F1, Y, ranks, permuted_F_hpart, row_selectors, si_groups)
    F0, D0 = F1, D1
    assert D1.min() >= -1e-8 #and loglikelihoods[-2] - 1e-8 <= loglikelihoods[-1]
    with open(output_path, 'wb') as handle:
        pickle.dump({"F":F1, "D": D1, "true_F": true_mfm.F, "true_D":true_mfm.D,
                     "ll":loglikelihoods[1:], "true_ll":true_ll, "hpart":F_hpart}, 
                     handle, protocol=pickle.HIGHEST_PROTOCOL)
    if t % 10 == 0:
        obj = mfm.fast_loglikelihood_value(F0, D0, Y, ranks, permuted_F_hpart, mfm_Sigma.num_factors(),
                                           tol1=1e-5, tol2=1e-5)
        loglikelihoods += [obj]
        print(f"{t=}, {obj=},  {D1.min()=}, {D1.max()=}")
