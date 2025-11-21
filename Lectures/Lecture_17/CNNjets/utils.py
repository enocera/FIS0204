import uproot3 as uproot
import numpy as np
import awkward as ak

# fix for XRootD/uproot4 issue: https://github.com/scikit-hep/uproot4/discussions/355


def get_file_handler(file_name):
    xrootd_src = file_name.startswith("root://")
    if not xrootd_src:
        return {"file_handler": uproot.MultithreadedFileSource}  # otherwise the memory maps overload available Vmem
    elif xrootd_src:
        # uncomment below for MultithreadedXRootDSource
        return {"xrootd_handler": uproot.source.xrootd.MultithreadedXRootDSource}
    return {}


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]


def to_np_array(ak_array, max_n=100, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, max_n, clip=True, axis=-1), pad).to_numpy()


def get_features_labels(file_name, features, spectators, labels, entry_stop=None):
    """
    Load features, labels, and spectator variables from a local ROOT file (uproot3-compatible).

    Parameters
    ----------
    file_name : str
        Path to local ROOT file.
    features : list of str
        Branch names for features.
    spectators : list of str
        Branch names for spectator variables.
    labels : list of str
        Branch names for labels.
    entry_stop : int or None
        Maximum number of entries to read. If None, read all.

    Returns
    -------
    feature_array : np.ndarray
        Feature array of shape (n_events, n_features)
    label_array : np.ndarray
        Label array of shape (n_events, 2)
    spec_array : np.ndarray
        Spectator array of shape (n_events, n_spectators)
    """

    # load file
    root_file = uproot.open(file_name)
    tree = root_file['deepntuplizer/tree']

    # read branches individually and slice to entry_stop
    N = entry_stop if entry_stop is not None else None

    feature_arrays = [tree.array(f)[:N] for f in features]
    spec_arrays = [tree.array(s)[:N] for s in spectators]
    label_arrays = [tree.array(l)[:N] for l in labels]

    # stack arrays
    feature_array = np.stack(feature_arrays, axis=1)
    spec_array = np.stack(spec_arrays, axis=1)
    label_array_all = {l: arr for l, arr in zip(labels, label_arrays)}

    njets = feature_array.shape[0]

    # construct label array
    label_array = np.zeros((njets, 2))
    label_array[:, 0] = label_array_all['sample_isQCD'] * (
        label_array_all['label_QCD_b'] +
        label_array_all['label_QCD_bb'] +
        label_array_all['label_QCD_c'] +
        label_array_all['label_QCD_cc'] +
        label_array_all['label_QCD_others']
    )
    label_array[:, 1] = label_array_all['label_H_bb']

    # remove unlabeled data
    mask = np.sum(label_array, axis=1) == 1
    feature_array = feature_array[mask]
    spec_array = spec_array[mask]
    label_array = label_array[mask]

    return feature_array, label_array, spec_array

def make_image(feature_array, n_pixels=224, img_ranges=[[-0.8, 0.8], [-0.8, 0.8]]):
    wgt = feature_array[:, 0]  # ptrel
    x = feature_array[:, 1]  # etarel
    y = feature_array[:, 2]  # phirel
    img = np.zeros(shape=(len(wgt), n_pixels, n_pixels))
    for i in range(len(wgt)):
        hist2d, xedges, yedges = np.histogram2d(x[i], y[i],
                                                bins=[n_pixels, n_pixels],
                                                range=img_ranges,
                                                weights=wgt[i])
        img[i] = hist2d
    return np.expand_dims(img, axis=-1)
