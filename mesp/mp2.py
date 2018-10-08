import numpy as np
import psi4

def do_mp2(mol):
    '''
    MP2 function
    
    Parameters
    ----------
    mol: MESP Molecule class
    '''
    
    mints = psi4.core.MintsHelper(mol.p4wfn.basisset()) # build a mints helper
    eri = np.asarray(mints.ao_eri()) # grab the 2-electron integrals in ao basis

    MO = np.einsum('ip,jq,kr,ls->pqrs',mol.C,mol.C,eri,mol.C,mol.C) # transform to MO basis
