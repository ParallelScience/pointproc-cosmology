# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_jackknife_covariance(data_matrix):
    n_realizations = data_matrix.shape[0]
    mean_val = np.mean(data_matrix, axis=0)
    jackknife_samples = []
    for i in range(n_realizations):
        sample = np.delete(data_matrix, i, axis=0)
        jackknife_samples.append(np.mean(sample, axis=0))
    jackknife_samples = np.array(jackknife_samples)
    cov = (n_realizations - 1) / n_realizations * np.sum((jackknife_samples - mean_val)**2, axis=0)
    return mean_val, cov

if __name__ == '__main__':
    data_dir = 'data/'
    n_realizations = 10
    vpf_list, k_list, j_list, mark_list = [], [], [], []
    for i in range(n_realizations):
        idx = str(i).zfill(2)
        vpf_list.append(np.load(os.path.join(data_dir, 'vpf_' + idx + '.npy')))
        k_list.append(np.load(os.path.join(data_dir, 'k_func_' + idx + '.npy')))
        j_list.append(np.load(os.path.join(data_dir, 'j_func_' + idx + '.npy')))
        mark_list.append(np.load(os.path.join(data_dir, 'mark_corr_' + idx + '.npy')))
    vpf_mean, vpf_var = compute_jackknife_covariance(np.array(vpf_list))
    k_mean, k_var = compute_jackknife_covariance(np.array(k_list))
    j_mean, j_var = compute_jackknife_covariance(np.array(j_list))
    mark_mean, mark_var = compute_jackknife_covariance(np.array(mark_list))
    print('VPF Variances:', vpf_var)
    print('K-function Variances:', k_var)
    print('J-function Variances:', j_var)
    print('Mark Correlation Variances:', mark_var)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].errorbar(np.arange(len(vpf_mean)), vpf_mean, yerr=np.sqrt(vpf_var), fmt='o-')
    axs[0, 0].set_title('Void Probability Function')
    axs[0, 1].errorbar(np.arange(len(k_mean)), k_mean, yerr=np.sqrt(k_var), fmt='o-')
    axs[0, 1].set_title('K-function')
    axs[1, 0].errorbar(np.arange(len(j_mean)), j_mean, yerr=np.sqrt(j_var), fmt='o-')
    axs[1, 0].set_title('J-function')
    axs[1, 1].errorbar(np.arange(len(mark_mean)), mark_mean, yerr=np.sqrt(mark_var), fmt='o-')
    axs[1, 1].set_title('Mark Correlation Function')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'summary_statistics.png'))
    print('Saved to ' + os.path.join(data_dir, 'summary_statistics.png'))