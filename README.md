# Visual Adversarial Imitation Learning using Variational Models (VMAIL)
This is the official implementation of the NeurIPS 2021 paper.

- [Project website][website]
- [Research paper][paper]
- [Datasets used in the paper][data]

[website]: https://sites.google.com/view/variational-mail
[paper]: https://arxiv.org/abs/2107.08829
[data]: https://drive.google.com/drive/folders/1JZmOVmlCqScqu0DDmn7857D5FtHZr6Un


## Method

![VMAIL](/images/VMAIL.png)

VMAIL simultaneously learns a variational dynamics model and trains an on-policy 
adversarial imitation learning algorithm in the latent space using only model-based 
rollouts. This allows for stable and sample efficient training, as well as zero-shot
imitation learning by transfering the learned dynamics model



## Instructions

Get dependencies:

```
conda env create -f vmail.yml
conda activate vmail
cd robel_claw/robel
pip install -e .
```

To train agents for each environmnet download the expert data from the provided link and run:

```
python3 -u vmail.py --logdir .logdir --expert_datadir expert_datadir
```

The training will generate tensorabord plots and GIFs in the log folder:

```
tensorboard --logdir ./logdir
```

## Citation

If you find this code useful, please reference in your paper:

```
@article{rafailov2021visual,
      title={Visual Adversarial Imitation Learning using Variational Models}, 
      author={Rafael Rafailov and Tianhe Yu and Aravind Rajeswaran and Chelsea Finn},
      year={2021},
      journal={Neural Information Processing Systems}
}
```