"""Exponential-random-graph models on constrained sample spaces.

The estimation core -- `sampler`, `mcmle`, `contrastive_divergence` and
`convex_hull` -- is not specific to any one model. It fits an ERGM whose
sample space is restricted by a constraint, using importance-sampled maximum
likelihood with the Hummel step length, seeded by contrastive divergence.

What makes that fast is specialising the sampler to the constraint rather than
proposing arbitrary tie toggles: when the constraint says what a valid
configuration looks like, a Gibbs move can be O(1) in the change statistics
instead of a network traversal.

`ergmpy.choice` is the first such constraint, the bipartite discrete choice
model of Sha et al. (2023), where each customer holds exactly one purchase
drawn from their own consideration set.
"""
