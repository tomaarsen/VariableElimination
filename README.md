# Variable Elimination
This repository holds an implementation of a Variable Elimination (VE) algorithm on Bayesian Networks, including a [paper](https://github.com/tomaarsen/VariableElimination/blob/main/paper.pdf) describing the algorithm, implementation, testing and more. The introduction of this paper is copied over to this README, and can be viewed below. There are two versions of the main file, `variable_elim.py` and `variable_elim_stripped.py`. The first version has more commenting and logging, while the latter is stripped down a tad.
The VE algorithm can be executed by running `run.py`.

---

# Introduction
Variable Elimination (VE) is an inference algorithm that can be applied on Bayesian Networks (BN), which efficiently sums out variables in a sensible order. It can be used to calculate probabilities of values for variables within such a Bayesian Network.

The algorithm for BN takes advantage of the BN property which states that

![\displaystyle P(X_1, \dots, X_n) = \prod_{i=1}^n P(X_i\ |\ Parents(X_i))](https://render.githubusercontent.com/render/math?math=\displaystyle%20P(X_1,%20\dots,%20X_n)%20=%20\prod_{i=1}^n%20P(X_i\%20|\%20Parents(X_i)))

where Parents($X_i$) is defined as all parents of node $X_i$ within the Bayesian Network.

This paper describes this algorithm, and implements it such that it can run on any given Bayesian Network. The BN's passed to this algorithm are in the [Bayesian Interchange Format](https://www.cs.washington.edu/dm/vfml/appendixes/bif.htm), samples of which can be found on the [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/) by `bnlearn`, a R package for Bayesian network learning and inference.

Alongside the explanation and implementation of this algorithm, four separate elimination ordering functions will be tested for performance. These functions are:
* `least_incoming_arcs`: Prioritises variables whose nodes in the BN have the **least parents**.
* `most_incoming_arcs`: Prioritises variables whose nodes in the BN have the **most parents**.
* `least_outgoing_arcs`: Prioritises variables whose nodes in the BN have the **least children**.
* `most_outgoing_arcs`: Prioritises variables whose nodes in the BN have the **most children**.

These functions are evaluated on computation time. How exactly these questions will be answered is described in the Methods section. Before getting to that, the Variable Elimination algorithm will be explained.

---

### Contributing
I am not taking contributions for this repository, as it is designed as an archive.

---

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.
