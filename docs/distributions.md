# Distributions
A package to do things for distributions that are either readily available elsewhere, or things that I haven't seen available elsewhere.

## Two Sided Power Distribution
The Two Sided Power Distribution (TSP) was introduced by van Dorp, and has been utilized in management engineering applications.

### moments
This functions provides the details of a given TSP distribution.  For example, you provide the parameters and it 
will return: E(x), Var, alpha, beta, p & q.  These parameters are calculated using van Dorps paper.<br><br> 
    <b>params:</b> LowBound, Middle, UpperBound, n <br>
    <b>results:</b> the values associated with the TSP, mean, variance, etc. 
<br><br>
``` moments(self, LowBound, Middle, UpperBound, n)```

#### Example
