import numpy as np
import psi4
psi4.core.be_quiet()
import mesp

geom = """
    O
    H 1 1
    H 1 1 2 104.5
    symmetry c1
"""
bas = "sto-3g"
mol = mesp.Molecule('H2O',geom,bas)

def test_mp2():
    mesp.mp2.do_mp2(mol)
    E_mesp = mol.E_MP2    

    psi4.set_options({
        'basis':bas,
        'mp2_type':'conv',
        'freeze_core':'false',
        'e_convergence':1e-12,
        'd_convergence':1e-12})
    psi4.set_module_options(
        'SCF', {
            'e_convergence': 1e-12, 
            'd_convergence': 1e-12,
            'DIIS': False,
            'scf_type':'pk'})
    E_psi4 = psi4.energy('MP2')
    
    print("Psi4 energy: {}\nmesp energy: {}".format(E_psi4,E_mesp))
    assert np.allclose(E_psi4,E_mesp)

if __name__=="__main__":
    test_mp2()
