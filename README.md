# GA-Reader
Code accompanying the paper [Gated Attention Reader for Text Comprehension](link).

## Prerequisites
- Python 2.7
- Theano and all dependencies (latest)
- Lasagne (latest)
- Numpy
- Maybe more, just use `pip install` if you get an error


## Preprocessed Data
You can get the preprocessed data files from [here](https://drive.google.com/drive/folders/0B7aCzQIaRTDUZS1EWlRKMmt3OXM?usp=sharing). Extract the tar files to the `data/` directory. Ensure that the symbolic links point to folders with `training/`, `validation/` and `test/` directories for each dataset.

## To run
Issue the command:
```
python run.py --dataset <wdw|cnn|dailymail|cbtcn|cbtne>
```

Complete list of options:
```
$ python run.py --help
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN 5105)
usage: run.py [-h] [--model MODEL] [--mode MODE] [--nlayers NLAYERS]
              [--dataset DATASET] [--seed SEED] [--gating_fn GATING_FN]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         base model (default: GAReader)
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
Bhuwan Dhingra and Hanxiao Liu

Report bugs and missing info to bdhingraATandrewDOTcmuDOTedu (replace AT, DOT appropriately).
