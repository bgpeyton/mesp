import numpy as np
import psi4
import mesp

def do_mp2(mol):
    '''
    MP2 function
    
    Parameters
    ----------
    mol: MESP Molecule class
    '''

    ### SETUP ###
    if not mol.scf_computed:
        mesp.do_scf(mol)
    
    # Make a Mints helper and grab AO integrals from psi4
    # Then transform them to MO basis
    mints = psi4.core.MintsHelper(mol.p4wfn.basisset()) 
    eri = np.asarray(mints.ao_eri()) 
#    MO = np.einsum('ip,jq,ijkl,kr,ls->pqrs',mol.C,mol.C,eri,mol.C,mol.C)

    ndocc = int(mol.nel / 2.) # number of doubly occ orbitals
    Co = mol.C[:,:ndocc]
    Cv = mol.C[:,ndocc:]
    MO = np.einsum('ip,jq,ijkl,kr,ls->pqrs',Co,Cv,eri,Co,Cv)

    o = mol.eps[:ndocc] # first ndocc orbital energies
    v = mol.eps[ndocc:] # virtual orbital energies

    D_ijab = 1 / (o.reshape(-1,1,1,1) + o.reshape(-1,1,1) - v.reshape(-1,1) - v) # similar to CCSD denominator

    E_1 = np.einsum('iajb,iajb,ijab->',MO,MO,D_ijab)
    E_2 = np.einsum('iajb,ibja,ijab->',MO,MO,D_ijab)

    E_MP2 = 2*E_1 - E_2 + mol.E_SCF
    mol.mp2_computed = True
    mol.E_MP2 = E_MP2
    print("MP2 energy computed!")
    print("MP2 energy = {}".format(E_MP2))
