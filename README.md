# sampling t-SNE

This repository contains my research into how applying the dimensional reduction algorithm
of t-SNE to a sample of a dataset influences the resulting embedding.
For this research, the quality of embeddings using different sampling algorithms were tested 
both numerically and visually.

### Repository Structure

The directory `utils` contains helper methods used in the analysis. This includes
implementations of the following sampling strategies: Furthest Point Sampling (`fps.py`),
Poisson Disk Sampling (`poisson_disk.py`) and Random Walk Sampling (`random_walk.py`). Additionally,
helper methods can be found in this directory. Quite a few of these were copied, with permission,
from research completed by Skrodzki et al. in the following research: https://arxiv.org/pdf/2308.15513. Specifically
this includes some indicated methods in `utils.py`, as well as all methods in `load_dataset.py` and `precision_recall.py`. 

The directory `src` contains the experiments actually run in the paper. This includes the following:
- `run_comparison_pipeline.py`: calculates embeddings and area under precision-recall curve for different sampling strategies. 
   This uses some helper methods which are defined in `visualization_helper.py`.
- `make_numerical_comparison_graphs.py`: make graph of area under precision-recall curve values
- `sub_clustering_visualization.py`: visualize datapoints for small clusters in poisson disk sampling