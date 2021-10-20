# BMANet: an efficient and chemically intuitive graph neural network for human oral bioavailability prediction

![message-passing](https://github.com/Y-Minghao/BMANet/blob/main/message-passing.png)
## requirements
    torch 1.7.1+cu110
    torch-cluster 1.5.8
    torch-scatter 2.0.5
    torch-sparse 0.6.8
    torch-spline-conv 1.2.1
    torch-geometric 1.6.3
    rdkit 2020.09.1.0
## human oral bioavailability dataset
The human oral bioavailability dataset requires an application to hou et al[1] for authorization to use, and I am not authorized to provide the human oral bioavailability dataset.
## BACE and BBBP
bbbp and BACE are datasets in Molecule net and benchmark commonly used in graph neural networks for molecular property prediction. The results on these two datasets are not presented in the paper in order not to detract from the central idea of the article - "human oral bioavailability".

    python run.py dataset_name gpu train_batch train_epoch
    
|  model  |  BBBP   |  BACE  |
|  ----  |  ----  | ----  |
| GCN | 0.882  | 0.851 |
| ARMA | 0.915  | 0.856 |
| MPNN | 0.902  | 0.848 |
| 单元格 | 单元格  | 单元格 |
| 单元格 | 单元格  | 单元格 |
| 单元格 | 单元格  | 单元格 |
| BMANet | 0.946  | 0.883 |

[1]Hou, Tingjun, et al. "ADME evaluation in drug discovery. 6. Can oral bioavailability in humans be effectively predicted by simple molecular property-based rules?." Journal of Chemical Information and Modeling 47.2 (2007): 460-463.
