# Primary-source constraints

- [Blanco-Pozo et al., Nature Neuroscience 2024](https://www.nature.com/articles/s41593-023-01542-x): reward-driven hidden-state changes survived dopamine perturbation; their reproducing model used recurrent cortex to predict observations and track hidden state, while a feed-forward basal-ganglia network learned value and policy with actor–critic RL.
- [Schuck et al., Neuron 2016](https://pubmed.ncbi.nlm.nih.gov/27657452/): human OFC represented hidden task states.
- [Vertechi et al., Neuron 2020](https://www.sciencedirect.com/science/article/pii/S089662732030043X): OFC inhibition causally shifted behavior away from hidden-state inference.
- [Bolkan et al., Nature Neuroscience 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5501395/): MD-to-mPFC supported maintenance and mPFC-to-MD supported later choice.
- [McNab and Klingberg, Nature Neuroscience 2008](https://www.nature.com/articles/nn2024): frontal and basal-ganglia activity preceded selective access to working memory.
- [Cui et al., Nature 2013](https://www.nature.com/articles/nature11846): direct- and indirect-pathway striatal populations were concurrently active during action initiation, contradicting a strict D1-only-Go/D2-only-NoGo switch.
- [Foster et al., Nature 2021](https://www.nature.com/articles/s41586-021-03993-3): CBGTC circuits contain parallel subnetworks, direct-path convergence, and closed recurrent loops.
- [Cavanagh et al., Nature Neuroscience 2011](https://pubmed.ncbi.nlm.nih.gov/21946325/): STN perturbation altered the relation between conflict and decision threshold.
- [Jin and Costa, Nature 2010](https://www.nature.com/articles/nature09263): striatal start/stop signals emerged during learned action sequences.
- [Jin, Tecuapetla, and Costa, Nature Neuroscience 2014](https://www.nature.com/articles/nn.3632): basal-ganglia subcircuits represented parsing and concatenation of action sequences.
- [Starkweather et al., Nature Neuroscience 2017](https://pubmed.ncbi.nlm.nih.gov/28263301/): dopamine RPEs were consistent with TD error over an inferred belief state; this does not show that dopamine computes the belief state.
- [Yagishita et al., Science 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4225776/) and [Shindou et al., European Journal of Neuroscience 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6585681/): corticostriatal activity can leave a short eligibility trace converted by later dopamine.

## Constraint, not analogy

The sources support a recurrent cortical state estimator, state-conditioned
striatal value/action channels, a conflict-sensitive execution threshold, local
three-factor credit, and learned action chunks. They do not establish XGBoost,
a fixed anatomical DAG, a Bayesian hazard bank inside BG, or an exact symmetric
D1/D2 learning rule.
