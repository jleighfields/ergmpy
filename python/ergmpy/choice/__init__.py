"""Bipartite discrete choice as a constrained ERGM.

Implements the model of Sha et al. (2023), "A network-based discrete choice
model for decision-based design," *Design Science* 9, e7. `constraints =
~b1degrees` with the consideration-set offset at `-Inf` restricts the sample
space to exactly one purchase per customer, which makes the conditional
distribution of any one customer's choice a multinomial over their own
alternatives.

Estimates are checked against the `ergm` R package's for the same data.
"""
