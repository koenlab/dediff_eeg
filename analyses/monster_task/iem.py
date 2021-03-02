# This is functions for IEM analysis

import numpy as np
from sklearn.utils.validation import column_or_1d


def make_basis_set(n_bins, centers, max=180):
    angles = np.arange(1,max+.25,.25)
    n_chans = len(centers)
    basis_set = []
    for c in centers:
        this_set = [np.cos(np.radians(a-c)) ** (n_chans-np.mod(n_chans,2)) for a in angles]
        basis_set.append(this_set)
    basis_set = np.array(basis_set).T
    return basis_set, angles

def make_c1_mat(labels, angles, basis_set):
    stim_mask = np.zeros([len(labels), len(angles)])
    for i in np.arange(stim_mask.shape[0]):
        stim_mask[i, labels.iloc[i] == angles] = 1
    c1 = np.matmul(stim_mask, basis_set)


@(xx,mu) (cosd(xx-mu)).^(n_ori_chans-mod(n_ori_chans,2));

%create basis set - assuming stimuli presented across all ranges of orientations
xx = linspace(1,180,180);
basis_set = nan(180,n_ori_chans);
chan_center = unique(stimlabels);%linspace(180/n_ori_chans,180,n_ori_chans);

for cc = 1:n_ori_chans
    basis_set(:,cc) = make_basis_function(xx,chan_center(cc));
end


class IEM(inst):
    inst = inst
    picks = 'all'
    @property
    def _get_bins(self):
        pass