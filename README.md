# online_cvxMF
This is the source code for paper "Online Convex Matrix Factorization with Representative Regions" 
https://arxiv.org/abs/1904.02580v2

`sample_data/` provides the synthetic data used for testing.

To run the algorithm with sample data, simply type in command line:

```shell
python -W ignore cvx_online_NMF.py --numIter <numIter> --NF 100 \
    --lmda 1e-5 \
    --dtype synthtetic_1 \
    --csize 500 \
    --candidate_size <candidate_size> \
    --pca -1
```
where `<numIter>` and `<candidate_size>` should be replaced with the desired number of iteration and size of representative set.

The command will run original MF, cvxMF, online DL and our purpose online cvxMF altogether and draw the plots with their clustering results.
