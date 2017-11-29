# GA-Reader
Code accompanying the paper [Gated Attention Reader for Text Comprehension](https://arxiv.org/abs/1606.01549).

## Prerequisites
- Python 2.7
- Theano (tested on 0.9.0dev1.dev-RELEASE) and all dependencies
- Lasagne (tested on 0.2.dev1)
- Numpy (>=1.12)
- Maybe more, just use `pip install` if you get an error


## Preprocessed Data
You can get the preprocessed data files from [here](https://drive.google.com/drive/folders/0B7aCzQIaRTDUZS1EWlRKMmt3OXM?usp=sharing). Extract the tar files to the `data/` directory. Ensure that the symbolic links point to folders with `training/`, `validation/` and `test/` directories for each dataset.

You can also get the pretrained Glove vectors from the above link. Place this file in the `data/` directory as well.

## To run
Issue the command:
```
python run.py --dataset <wdw|cnn|dailymail|cbtcn|cbtne>
```

Complete list of options:
```
$ python run.py --help
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN 5105)
usage: run.py [-h] [--mode MODE] [--nlayers NLAYERS] [--dataset DATASET]
              [--seed SEED] [--gating_fn GATING_FN]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           run mode - (0-train+test, 1-train only, 2-test only,
                        3-val only) (default: 0)
  --nlayers NLAYERS     Number of reader layers (default: 3)
  --dataset DATASET     Dataset - (cnn || dailymail || cbtcn || cbtne || wdw)
                        (default: wdw)
  --seed SEED           Seed for different experiments with same settings
                        (default: 1)
  --gating_fn GATING_FN
                        Gating function (T.mul || Tsum || Tconcat) (default:
                        T.mul)
```

To set dataset specific hyperparameters modify `config.py`.

## Note
Make sure to add `THEANO_FLAGS=device=cpu,floatX=float32` before any command if you are running on a CPU.

## Contributors
If you use this code please cite the following:

Dhingra, B., Liu, H., Yang, Z., Cohen, W. W., & Salakhutdinov, R. (2016). Gated-Attention Readers for Text Comprehension. arXiv preprint arXiv:1606.01549.
```
@article{dhingra2016gated,
  title={Gated-Attention Readers for Text Comprehension},
  author={Dhingra, Bhuwan and Liu, Hanxiao and Yang, Zhilin, and Cohen, William W and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:1606.01549},
  year={2016}
}
```

Report bugs and missing info to bdhingraATandrewDOTcmuDOTedu (replace AT, DOT appropriately).
