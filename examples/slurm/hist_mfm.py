import numpy as np
import random

import timeit, warnings, argparse, os

import mfmodel as mfm
import numba as nb
import pickle




# Ignore specific warning
warnings.filterwarnings("ignore", message="omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.")

# Ignore NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=nb.NumbaPerformanceWarning)


parser = argparse.ArgumentParser(description='Solve a problem based on rank and n.')
parser.add_argument('--slurm', type=int, default=0)
args = parser.parse_args()

n = 10000

np.random.seed(1001 + args.slurm)
random.seed(1001 + args.slurm)

signal_to_noise = 4

L = 6

mtype = "large_smfm_n%d_L%d"%(n, L)
filename = "%s_n%d_sl%d"%(mtype, n, args.slurm)

ranks = np.array([10, 5, 4, 3, 2, 1])
rank = ranks.sum()
nsamples = 4 * rank
nsamples



pi_rows = np.random.permutation(n)
hpart = {'rows':{'pi':pi_rows, 'lk':[]}, 'cols':{'pi':pi_rows, 'lk':[]}} 
for ngroups in [2, 5, 9, 17, 33, n+1]:
       hpart['rows']['lk'] += [ np.linspace(0, n, ngroups, endpoint=True, dtype=int)]
hpart['cols']['lk'] = hpart['rows']['lk']
part_sizes = mfm.print_hpart_numgroups(hpart)
mfm.valid_hpart(hpart)



true_mlr, true_sparse_F, true_D_noise = mfm.generate_mlr_model(n, hpart, ranks, signal_to_noise)
row_selectors, si_groups, F_hpart = mfm.row_col_selections(hpart)
permuted_F_hpart = {"pi_inv":np.arange(n), "pi":np.arange(n), "lk":F_hpart["lk"]}

print(f"{n=}, {nsamples=}, {L=}, {ranks.sum()=}")


C = mfm.generate_data(true_sparse_F, true_D_noise, nsamples, true_mlr)
Z = (C - C.mean(axis=1, keepdims=True))[hpart["rows"]["pi"], :]
del C
unpermuted_A = (Z @ Z.T / (Z.shape[1]-1))[true_mlr.pi_inv_rows, :][:, true_mlr.pi_inv_cols]
# permute to put clusters on diagonal
Y = Z.T
N = Y.shape[0]

true_F, true_D = true_mlr.B[:, :-1]+0, true_D_noise+0
true_mfm = mfm.MFModel(F=true_F, D=true_D, hpart=F_hpart, ranks=ranks)

true_train_obj = mfm.fast_loglikelihood_value(true_F, true_D, Y, ranks, permuted_F_hpart,
                                           tol1=1e-5, tol2=1e-5)
exp_true_ll = mfm.fast_exp_true_loglikelihood_value(true_F, true_D, ranks, F_hpart,
                                           tol1=1e-5, tol2=1e-5)
print(f"TR: train ll={true_train_obj}, exp ll={exp_true_ll}")


ll_distribution = {"frob":{"train":[], "exp":[]},
                   "mle":{"train":[], "exp":[]}}



script_dir = "/home/groups/boyd/tetianap/mfm"
output_path = os.path.join(script_dir, 'outputs/%s.pickle' % filename)


for t in range(25):
    print(f"{t=}")
    C = mfm.generate_data(true_sparse_F, true_D_noise, nsamples, true_mlr)
    Z = (C - C.mean(axis=1, keepdims=True))[hpart["rows"]["pi"], :]
    del C
    unpermuted_A = (Z @ Z.T / (Z.shape[1]-1))[true_mlr.pi_inv_rows, :][:, true_mlr.pi_inv_cols]
    # permute to put clusters on diagonal
    Y = Z.T
    N = Y.shape[0]

    # Frobenius 
    frob_mfm, losses = mfm.fast_frob_fit_loglikehood(unpermuted_A, Y, F_hpart, hpart, ranks, printing=False, eps_ff=1e-3)
    frob_mfm.D = np.maximum(1e-6, frob_mfm.D )
    obj_frob = mfm.fast_loglikelihood_value(frob_mfm.F, frob_mfm.D, Y, ranks, permuted_F_hpart)
    frob_mfm.inv_coefficients()
    obj_frob_exp = mfm.fast_exp_loglikelihood_value(np.concatenate([true_F, np.sqrt(true_D).reshape(-1, 1)], axis=1), 
                                                            frob_mfm, ranks, hpart["rows"], F_hpart, 
                                                            row_selectors, si_groups, tol1=1e-8, tol2=1e-8)
    
    print(f"FR: train ll={obj_frob}, exp ll={obj_frob_exp}")
    ll_distribution["frob"]["train"] += [obj_frob]
    ll_distribution["frob"]["exp"] += [obj_frob_exp]

    # MLE
    fitted_mfm, loglikelihoods = mfm.fit(Y, ranks, F_hpart, printing=False, max_iter=300, freq=100)
    fitted_mfm.inv_coefficients()
    obj_exp = mfm.fast_exp_loglikelihood_value(np.concatenate([true_F, np.sqrt(true_D).reshape(-1, 1)], axis=1), 
                                                            fitted_mfm, ranks, hpart["rows"], F_hpart, 
                                                            row_selectors, si_groups, tol1=1e-8, tol2=1e-8)

    print(f"ML: train ll={loglikelihoods[-1]}, exp ll={obj_exp}")
    ll_distribution["mle"]["train"] += [loglikelihoods[-1]]
    ll_distribution["mle"]["exp"] += [obj_exp]
    if t % 5:
        print(- np.mean(ll_distribution["frob"]["exp"]) + np.mean(ll_distribution["mle"]["exp"]))

    with open(output_path, 'wb') as handle:
        pickle.dump(ll_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)

