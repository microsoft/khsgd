# KH-SGD

**Kernel Halving SGD** (KH-SGD) iteratively reorders datapoints during stochastic gradient descent training to provably accelerate convergence.

For a detailed description of the KH-SGD algorithm and its guarantees, see [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063).

```bib
@inproceedings{carrell2025lowrank,
  title={Low-Rank Thinning},
  author={Annabelle Michael Carrell and Albert Gong and Abhishek Shetty and Raaz Dwivedi and Lester Mackey},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=iAkg2nVmvN}
}
```

This codebase reproduces the reordered SGD experiments of [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063) and is derived from the code of [CD-GraB](https://github.com/GarlGuo/CD-GraB).

## Dependencies
This code has been tested with the following operating system, Python, and PyTorch combinations:
- Rocky 8.9, Python 3.10, Torch 2.6.0

The following dependences are needed to run the experiment:
- Python >= 3.9
- PyTorch >= 2.0.0
- CUDA >= 11.7 on linux
- torchopt
- torchvision
- functorch
- transformers

Below are step by step commands to create a Conda environment with the proper dependencies:
```
conda create -n khsgd python=3.10
conda activate khsgd
# Follow instructions at https://pytorch.org/get-started/locally/ for your system
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install torchopt
conda install functorch
conda install transformers
```

After, download the [HMDA preprocessed files](https://github.com/GarlGuo/CD-GraB/tree/main/data/HMDA) and place them under `data/HMDA/`.

## Experiments

To recreate the reordered SGD experiments of [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063), please run
```
torchrun --nproc_per_node=1 --nnodes=1 --master_addr="localhost" --master_port=35500 main-LR-HMDA.py --sorter <SORT> --seed <SEED> --lr 5e-3 --node_cnt 1
```
with `<SORT>` replaced by each of the sorters ("CD-GraB", "D-RR", "KH-SGD", "SBW") and `<SEED>` replaced by each seed in the range 1-5.

To recreate the plot, run `python LR-HMDA.py` in the [notebooks/LR-HMDA](notebooks/LR-HMDA) directory.

To recreate the supplementary plot, add `torch.save(gathered_grads, f"PREFIX_{batch}")` to line 291 in `d_hmda.py`. Run the main script as described above. Then, run [notebooks/LR-HMDA/sing_vals.ipynb](notebooks/LR-HMDA/sing_vals.ipynb), with `PREFIX` as `FILENAME_PREFIX` (by default it is `all_grads_epoch`) appended to your data location.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
