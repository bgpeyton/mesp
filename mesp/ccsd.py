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
    if not mol.scf_computed:
        mesp.do_scf(mol)

    # Make a Mints helper and grab integrals from psi4
    # Alternatively, could grab the AO eri, transform them to MO
    # (as in MP2), then expand them to spin-orbitals: 
    # for n in range(1,5):
    #   eri = np.repeat(eri,2,axis=n) 
    # then remove the zero-by-symmetry integrals
    # and finally anti-symmetrize
    #
    # Alt-alternatively, the SO transformation can be done
    # by kronecker multiplying MO-eri by a 4D (2,2,2,2)
    # "identity" which expands and spin-zeroes at the same time
    # eri = np.kron(eri,I)
    # then antisymmetrize
    mints = psi4.core.MintsHelper(mol.p4wfn.basisset())
    C_p4 = psi4.core.Matrix.from_array(mol.C) # need p4 Matrix
    MO = np.asarray(mints.mo_spin_eri(C_p4,C_p4)) # antisymmetrized SO MO ERI

    nso = MO.shape[0] # convenient number of spin orbitals
    nocc = int(mol.nel) # number of occ orbitals
    nvir = nso - nocc # number of vir orbitals
    so_eps = np.repeat(mol.eps, 2) # spin-orbital sfc orbital energies

    F = mol.Hc + 2*mol.J - mol.K # AO-basis Fock
    F = np.einsum('uj,vi,uv', mol.C, mol.C, F) # MO-basis Fock
    I = np.array([[1,0],
                  [0,1]])
    F = np.kron(F,I) # Expand and spin-tegrate

    # use slices to cut out occ and vir blocks
    o_slice = slice(0,nocc) 
    v_slice = slice(nocc,nso)

    MO_ijab = MO[o_slice,o_slice,v_slice,v_slice] # <ij||ab>

    # Grab orbital energies. I know I could just tile mol.eps,
    # but this is a good check if my Fock is correct
    diag = np.diag(F) # orbital eigenvalues in a 1D array
    F_occ = diag[:nocc] # grab first nocc diag elements
    F_vir = diag[nocc:] # grab the last nvir elements

    # Clever construction of denominator matrices- thanks Ashutosh!
    D_ia = F_occ.reshape(-1,1) - F_vir # Used for t1
    D_ijab = F_occ.reshape(-1,1,1,1) + F_occ.reshape(-1,1,1) - F_vir.reshape(-1,1) - F_vir # Used for t2

    t1 = 0 # initial t1 guess

    t2 = MO_ijab / D_ijab # initial t2 guess

    E_MP2_CORR = 0.25 * np.einsum('ijab,ijab->',MO_ijab,t2)
    E_MP2 = E_MP2_CORR + mol.E_SCF
    print('E_MP2 = {}'.format(E_MP2))
