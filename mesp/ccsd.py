import numpy as np
import psi4

def do_ccsd(mol):
    '''
    CCSD function
    
    Parameters
    ----------
    mol: MESP Molecule class

    '''
    
    ### SETUP ###
    # Make a Mints helper and grab integrals from psi4
    # Alternatively, could grab the AO eri, transform them to MO
    # (as in MP2), then expand them using something like 
    # for n in range(1,5):
    #   eri = np.repeat(eri,2,axis=n) 
    # and then remove the zero-by-symmetry integrals
    mints = psi4.core.MintsHelper(mol.p4wfn.basisset())

    MO = mints.mo_spin_eri(mol.C,mol.C)

