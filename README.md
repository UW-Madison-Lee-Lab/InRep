


### Structure of the source code:

- src: 
  - configs: configurations 
  - evals: evaluation scripts
  - metrics: metrics using in the paper
  - models: all GAN conditoining models and networks 
  - scripts: bash scripts to run
  - utils: helper functions
  - constant.py: define all configuration names
  - train.py: training function
  - main.py: main entry point
- results:
  - checkpoints: saved models
  - evals: log file of results

### Running script
```
python main.py --data_type <data>   --gan_type <method> --exp_mode <experiment>  --label_ratio <label_ratio>
```

Here, values of arguments can be found in constant.py 

### Training

We use the the main script with flag: --is_train. For instance, to train InRep+ on CIFAR10 with 10% supervision, we use the script:

```
python main.py --data_type  cifar10   --gan_type inrep --exp_mode complexity --label_ratio 0.1 --is_train --nepochs 10 ---nsteps_save 10
```
### Inference

For testing, we use the main script with testing metrics (fid, intrafid, cas, pr). For instance, to test FID of  InRep+ on CIFAR10 with 10% supervision, we use the following script:

```
python main.py --data_type  cifar10   --gan_type inrep --exp_mode complexity --label_ratio 0.1 -t fid
```
### Credits

We reuse the repositories from several sources: [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN), [rGAN](https://github.com/takuhirok/rGAN), [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

