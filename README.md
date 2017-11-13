# lava

nonlinearSS.py generates data from a 2x2 system and identifies the parameters. 
The parameters can be accessed using 
```python
estimate.Theta 
estimate.Z
```

To simulate the identified system use
```python
Y_sim = estimate.simulate(U)
```
