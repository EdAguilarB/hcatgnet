# HCat-GNet (Homogeneous Catalyst Graph Neural Network)

The **HCat-GNet** (Homogeneous Catalyst Graph Neural Network) is an open-source platform intended for virtual evaluation of homogeneous catalysts selectivity. In principle, the software presented in this repository should be useful and accurate to predict any target property that wants to be optimised in a process accelerated by an homogenous catalyst. 

Rh RhCAA directory contains the results reported and code used to get the results reported in the paper: **HCat-GNet: An Interpretable Graph Neural Network for Catalysis Optimization**

In order to be able to run the experiments presented in the paper, you must create an environment using the yml file provided.

To reproduce all the experiments produced with the default setting, run the command:

<pre>
  <code>
    python run_experiments.py
  </code>
</pre>

To run the experiments using the CircuS descriptors, run:

<pre>
  <code>
    python run_experiments.py --descriptors circus
  </code>
</pre>
