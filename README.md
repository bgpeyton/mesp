### MESP: My Electronic Structure Package
A Psi4NumPy-based electronic structure package for test and reference implementations in Python.
Same requirements as [Psi4NumPy](https://github.com/psi4/psi4numpy). 

To install, run:
```python
pip install -e .
```

To test, run:
```python
py.test
```

### Implemented methods:
* SCF
* * RHF (with DIIS)
* MP2
* * Spin-orbital
* CCSD (spin-adapted and spin-orbital)
* * Spin-orbital
* * Spin-adapted (with DIIS)
* CC2  (spin-adapted and spin-orbital)
* * Spin-orbital
* * Spin-adapted (with DIIS)
