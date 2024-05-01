import numpy as np
import random

import timeit, warnings, argparse, os

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

pi_rows = np.random.permutation(n)
hpart = {'rows':{'pi':pi_rows, 'lk':[]}, 'cols':{'pi':pi_rows, 'lk':[]}} 

for ngroups in [2, 5, 9, 17, 33, n+1]:
       hpart['rows']['lk'] += [ np.linspace(0, n, ngroups, endpoint=True, dtype=int)]
hpart['rows']['lk'][1] = np.delete(hpart['rows']['lk'][1], -2)
hpart['rows']['lk'][2] = np.delete(hpart['rows']['lk'][2], -4)
hpart['cols']['lk'] = hpart['rows']['lk']
part_sizes = mfm.print_hpart_numgroups(hpart)
mfm.valid_hpart(hpart)



F_hpart = {"pi": hpart['rows']["pi"], "lk": hpart['rows']["lk"][:-1]}
true_mfm = mfm.MFModel()
true_mfm = mfm.generate_mfmodel(true_mfm, n, F_hpart, ranks, signal_to_noise, debug=False)
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
true_ll = {"train":true_train_obj, "exp":exp_true_ll}
print(f"True: {true_ll}")


mfm_Sigma = mfm.MFModel(hpart=F_hpart, ranks=ranks)
mfm_Sigma.init_FD(ranks, F_hpart, init_type="Y", Y=Y)
F0, D0 = mfm_Sigma.F, mfm_Sigma.D


script_dir = os.getcwd()
# script_dir = "/home/groups/boyd/tetianap/mfm"
# parent_dir = os.path.dirname(script_dir)
output_path = os.path.join(script_dir, 'outputs/%s_rank%d_L%d.pickle' % (mtype, rank, L))


loglikelihoods = [-np.inf]
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
