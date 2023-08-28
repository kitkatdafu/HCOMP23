# Part a
- `[NAME]_t.csv`
   - This file contains the number of times each pair is queried.
   - Each row corresponds to a pair. 
   - The first column is the id of the image on the left of the pair.
   - The second column is the id (cluster id) of the image (cluster) on the right of the pair.
   - The third column is the number of times this pair is queried.
- `[NAME]_time.csv`
   - Each row of this file corresponds to a worker.
   - Each row is a list of time that the worker took to complete a query.
   - The unit of time is milliseconds.
- `[NAME]_worker_error_rate.csv`
    - Each row of this file corresponds to a worker.
    - The first column is the number of queries answered by the worker.
    - The second column is the number of queries answered correctly by the worker.
# Part b
- `our_K3` and `our_K30` contains the simulation results of our algorithm using the simulated dataset (K=3 and K=30).
   - VI: variation of information
   - mean_T: mean number of queries per pair
   - total queries: total number of queries
   - pair error rate: pair error rate
   - K: predicted K
- `yun14_passive_K{}_diffT.csv` contains the simulation results of yun14 passive.
   - VI_yun: variation of information
   - edge_error_rate_yun: edge error rate
   - total_queries: total number of queries
   - K: ground truth number of clusters
   - K_predicted: predicted number of clusters
   - T: budget (number of queries)
   - repeats per pair: number of times each each pair is sampled
- `yun14_simulation_K{}.csv` contains the simulation results of yun14 active.
   - T: budget (number of queries)
   - K: ground truth number of clusters
   - VI: variation of information with budget overflow
   - VI_r: variation of information without budget overflow
   - edge_error_rate: edge error rate with budget overflow
   - edge_error_rate_r: edge error rate without budget overflow
   - K_predicted: predicted number of clusters
   - T_actual: total number of queries (with budget overflow)
- `yun14_adaptive_allsports.csv` contains the allsports results of yun14 active. The columns are similar to the ones in
  `yun14_simluation_K{}.csv`.
- `yun14_passive_allsports_diff_T.csv` contains the allsports results of yun14 passive:
   - VI_yun: variation of information
   - edge_error_rate_yun: edge error rate
   - K_predicted: predicted number of clusters
   - T: budget (number of queries)
   - mean_T: mean number of queries per pair


## Note on budget overflow
While running the Yun14 adaptive, if the budget T is reached, we do not terminate the algo immediately. We pretend that there are still budgets remaining and cluster the remaining nodes. If we let C denote this clustering result, then T_actual is the actual budget corresponding to C, and VI/edge_error_rate is the corresponding performance measures.
Suppose O is the set of nodes that got clustered after the budget is used up. Then after obtaining C, We remove O from C and assign them back to C at randomly selected clusters. The VI_r and edge_error_rate_r are the corresponding measures.