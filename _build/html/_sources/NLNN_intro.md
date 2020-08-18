# What is NLNN?

![Non-local neural network](NLNN.png)

- Non-local neural network (**NLNN**) is a variant of Graph Network.
- In our model, the **NLNN** takes as *input a graph* with node and edge attributes and *outputs a graph* with updated node attributes.

**NLNN algorithm**
1. For each edge $k$ in the input graph,  

    - compute updated edge attributes  
    $\textbf{e}_{k}^{\prime} = \phi^{e} \left( \left[ \textbf{v}_{s_k}, \textbf{v}_{r_k} \right] \right)$,  
    where $\textbf{v}_{s_k}, \textbf{v}_{r_k}$ are sender and receiver nodes attributes resp. of edge $k$,  
    $\left[ \cdot, \cdot \right]$ denotes concatenation of attributes.  
    
2. For each vertex $i$ in the input graph,  

    - aggregate attributes of all edges adjacent to node $i$  
    $\bar{\textbf{e}}_{i}^{\prime} = \rho^{e \rightarrow v} \left( \textbf{E}_{i}^{\prime} \right)$,  
    where $\textbf{E}_{i}^{\prime}$ is set of received and sent edge attributes.  
    
    - compute updated node attributes    
    $\textbf{v}_{i}^{\prime} = \phi^{v} \left( \left[ \textbf{v}_{i}, \bar{\textbf{e}}_{i}^{\prime} \right] \right)$.  
    
**Update functions**
1. $\phi^{e}$: Single Layer Perceptron with output attribute size 16.  
2. $\phi^{v}$: Single Layer Perceptron with output attribute size 16.  

**Aggregate functions**
1. $\rho^{e \rightarrow v}$: summation.
