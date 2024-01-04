
# earthPT

<p align="center">
    <img src="assets/emoji.png" alt="earthPT" width="250"/>
</p>

A simple repository for training time series large observation models. This
repository began its life as Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT), and has been altered so that
it is usable for time series data. `train.py` reproduces
[EarthPT-700M](https://arxiv.org/abs/2309.07207) when trained on 14B time
series 'tokens' of ClearSky EO data. When `train.py` takes ~5 days to run to
Chinchilla completion on a single 8xA100 40GB node.
`train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line
GPT model definition with an MLP tokeniser and a regressive loss.  That's it.

We have purposefully kept the code as simple and hackable as possible so that
it is easy for all to hack this base to their needs. We release this code under
the MIT licence in the hope that it will prove useful to others working on
EO and timeseries large observation models.

I will populate this readme with docs on usage in the coming weeks.

## install

Dependencies:

- `pip install -r requirements.txt`
