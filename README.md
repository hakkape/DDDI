# Interval-based Dynamic Discretization Discovery for solving the Continuous-Time Service Network Design Problem

This repository contains some of the code and instances associated with the [DDDI paper published in Transportation Science (2020).](https://doi.org/10.1287/trsc.2020.0994)



## Benchmark Results

The following results were produced on a desktop computer with an AMD Ryzen Threadripper PRO 5945WX 12-core CPU and 64 GB of RAM. Termination with 1% gap or after 1 hour. Solver is Gurobi v12.0.3.

The "Solve (s)" only includes time spent in Gurobi, whereas "Time (s)" includes everything.  As you can see the python code has relatively low overhead, but could be improved.

### Standard Instances

| Class  | Gap (%) | Time (s) | Solve (s) | # its | Solved (%) |
|-------:|--------:|---------:|----------:|------:|-----------:|
| HC/HF  | 1.04%   | 576.91   | 568.36    | 17.8  | 92.7%      |
| HC/LF  | 0.83%   | 61.22    | 51.96     | 11.4  | 100.0%     |
| LC/HF  | 0.55%   | 0.04     | 0.01      | 0.5   | 100.0%     |
| LC/LF  | 0.69%   | 0.45     | 0.15      | 2.9   | 100.0%     |


### Significant Time Points

Uses initial time points as per ["New Dynamic Discretization Discovery Strategies
 for Continuous-Time Service Network Design"](https://optimization-online.org/wp-content/uploads/2025/01/enhanced_DDD_20250116.pdf).  Results do not include time taken to find initial time points (only takes a few seconds typically). Also, uses "reduced" algorithm instead of "adaptive" (i.e., does not adaptively change gap termination for the IP).

| Class  | Gap (%) | Time (s) | Solve (s) | # its | Solved (%) |
|-------:|--------:|---------:|----------:|------:|-----------:|
| HC/HF  | 0.79%   | 392.65   | 391.01    | 0.5   | 96.0%      |
| HC/LF  | 0.41%   | 35.99    | 34.60     | 0.1   | 100.0%     |
| LC/HF  | 0.12%   | 0.07     | 0.05      | 0.0   | 100.0%     |
| LC/LF  | 0.15%   | 0.43     | 0.32      | 0.0   | 100.0%     |


### SND-RR

| Class           | Gap (%) | Time (s) | Solve (s) | # its | Solved (%) |
|----------------:|--------:|---------:|----------:|------:|-----------:|
| critical_times  | 0.73%   | 17.87    | 12.49     | 22.7  | 100.0%     |
| designated_path | 0.80%   | 26.17    | 11.13     | 31.6  | 100.0%     |
| hub_and_spoke   | 0.62%   | 6.73     | 3.88      | 17.1  | 100.0%     |


## Code Information
The original code is quite old, written during my PhD (2014 - 2017).  Several changes have been made to update to Python 3.x and improve code quality. Also, I've recently added support for reading instances from [SND-RR](https://github.com/madisonvandyk/snd-rr).

I never originally intended to release the code, so it has many hacks and is quite ugly! But for my own peace-of-mind I plan to slowly clean it up.

That said, it seems like there is ongoing research into DDD, and this code is still quite competitive with state-of-the-art. So it might be useful for comparisons.

## Reference

If you use this, I'd love to hear about it. For academic usage, please cite this article.

```
Luke Marshall, Natashia Boland, Martin Savelsbergh, Mike Hewitt (2020) Interval-Based Dynamic Discretization Discovery for Solving the Continuous-Time Service Network Design Problem. Transportation Science 55(1):29-51.
https://doi.org/10.1287/trsc.2020.0994
```