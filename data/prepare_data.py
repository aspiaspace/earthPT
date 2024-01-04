"""
This file translates clearsky to sequences ready for our eogpt model to train on.

2023-02 mike.smith@aspiaspace.com
"""
import numpy as np
import h5py as h5
import re
import sys
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

BASENAME = "TL_CS"

def get_h5(fname, next_fname):
    """
    Convert h5 file to numpy array.
    """
    # THIS ASSUMES ONLY ONE INSTANCE OF DATE!!
    def _get_embedding(_fname):
        toy = re.search('....-..-..\.h5', _fname.name).group(0)[:-3]
        toy = datetime.strptime(toy, '%Y-%m-%d')
        day = (toy - datetime(toy.year, 1, 1)).days
        embedding = (np.sin((day/365)*2*np.pi), np.cos((day/365)*2*np.pi))
        return embedding

    embedding = _get_embedding(fname[0])
    next_embedding = _get_embedding(next_fname[0])

    with h5.File(fname[0], "r") as f:
        cs = np.stack([f[fname[0].name[:4]][ch] 
            for ch in [
                "Blue",
                "Green",
                "Red",
                "Red Edge 1",
                "Red Edge 2",
                "Red Edge 3",
                "NIR",
                "Red Edge 4",
                "SWIR 1",
                "SWIR 2",
            ]])

    days_sin = np.ones_like(cs[:1])*embedding[0]
    days_cos = np.ones_like(cs[:1])*embedding[1]
    next_days_sin = np.ones_like(cs[:1])*next_embedding[0]
    next_days_cos = np.ones_like(cs[:1])*next_embedding[1]
    return np.concatenate([(cs/1000)*2-1, days_sin, days_cos, next_days_sin, next_days_cos], axis=0)

if __name__ == "__main__":
    grid = int(sys.argv[1])
    tile = f"TL{grid:02d}"

    for split in ["train", "test"]:
        fnames = np.array(sorted(Path(f"./TL_CS/").glob(f"{tile}_clearsky_????-??-??.h5")))
        dates = np.array(list(map(
            lambda f: datetime.strptime(re.search('....-..-..\.h5', f.name).group(0)[:-3], '%Y-%m-%d'),
            fnames,
        )))

        if split == "train":
            train_dates = dates < datetime.strptime('2022-12-31', '%Y-%m-%d')
            fnames_cs = fnames_cs[train_dates]
            fnames_sar = fnames_sar[train_dates]
        else:
            test_dates = dates > datetime.strptime('2022-12-31', '%Y-%m-%d')
            fnames_cs = fnames_cs[test_dates]
            fnames_sar = fnames_sar[test_dates]

        print(f"{tile}: processing {len(fnames)} frames")

        bigar = np.zeros((1024*1024, len(fnames)-1, 18))
        for i, fname, next_fname in tqdm(zip(
                                      range(len(fnames)-1), 
                                      fnames,
                                      fnames[1:]),
                                      total=len(fnames)-1
                                    ):
            bigar[:, i] = np.swapaxes(get_h5(fname, next_fname).reshape(18, 1024*1024), 0, 1)

        np.save(f"./TL_EOPT/{tile}_{split}.npy", bigar.astype(np.float16))
