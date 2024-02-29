# Crowdsourced Clustering via Active Querying: Practical Algorithm with Theoretical Guarantees

![Intro Figure](illusration.png)

This repository provides code for the simulations and experiments in our paper: [Crowdsourced Clustering via Active Querying: Practical Algorithm with Theoretical Guarantees](https://www.deepneural.network/publications/pdfs/hcomp23.pdf) @ The Eleventh AAAI Conference on Human Computation and Crowdsourcing (HCOMP 2023).

## Structure
The root of the codebase contains 5 directories.
1. `backend` contains code for the backend (removed in revised version due to space limits).
2. `csv` contains data in `csv` format, generated from the experiments run using the backend.
3. `mat` contains data in `mat` format, generated from the experiments run using the backend and simulation.
4. `notebook` contains files that used to visualize data obtained from the experiments.
5. `simulation` contains code to generate simulated results.

## Environment
### Simulation
The required packages and their corresponding packages are located in `environment.yml`. You first need to make sure the
required packages are installed. One easy way to do so is to use `Anaconda` to create an environment directly:
```
conda env create -f environment.yml
```
This will create an environment named `active-querying`. To active this environment, run
```
conda activate active-querying
```

## Usage
### Simulation
The files related to the simulation are located in the `simulation` directory.
There are two python files whose name starting with `executor`. These two files are the entry point for the simulation.

- The file `executor_all_sports.py` runs simulation using the results obtained from the all sports experiment.
- The file `executor_yun_14.py` runs simulation on the simulated dataset that was created for the yun 14 simulation.
- The file `executor.py` runs the simulation in general.
- Outputs of the simulation are stored in the `outputs` directory.
- Directory `yun14-related` contains files related to simulations regarding yun14 algorithm.
  - `clusering.py`  contains the implementation of yun14.
  - `passive_simulation[].ipynb` are used to generate adjacency matrix, frequency matrix, or observation matrix for
    simulation.
  -  `yun14_passive_all[].ipynb` runs the yun14 passive simulation.
  - `adpative_simulation.ipynb` runs the yun14 adaptive on simulated dataset.
  - `adpative_allsports.ipynb` runs the yun14 adaptive on allsports dataset.

The parameters related to the simulation can be set in the `main()` function in the two files.

## Citation

If you find our repository useful for your research, please consider citing our paper:
```
@inproceedings{chen2023crowdsourced,
  title={Crowdsourced Clustering via Active Querying: Practical Algorithm with Theoretical Guarantees},
  author={Chen, Yi and Vinayak, Ramya Korlakai and Hassibi, Babak},
  booktitle={Proceedings of the AAAI Conference on Human Computation and Crowdsourcing},
  volume={11},
  number={1},
  pages={27--37},
  year={2023}
}
```

