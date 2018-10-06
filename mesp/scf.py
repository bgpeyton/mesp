import numpy as np
import psi4

def diag(A,F):
    '''
    Diagonalize a Fock matrix F using diagonalization matrix A
    Returns Eigenvalues e and Eigenfunctions C (in original basis)
    '''
    Fd = A.T @ F @ A # Diagonalize F
    e, C = np.linalg.eigh(Fd) # Solve for eigenvalues and eigenvectors
    C = np.einsum('ik,kj->ij',A,C) # Return C to nonorthogonal AO basis
    return e, C

def do_scf(mol,
        e_conv = 1e-12,
        d_conv = 1e-12,
        max_iter = 50):
    '''
    SCF function
    
    Parameters
    ----------
    mol: MESP Molecule class
    e_conv: float
    d_conv: float
    max_iter: int
    '''
    
    ### SETUP ###
    # Make a Mints helper and grab integrals from psi4
    mints = psi4.core.MintsHelper(mol.p4wfn.basisset())

    S = np.asarray(mints.ao_overlap())   # AO basis overlap 
    T = np.asarray(mints.ao_kinetic())   # AO basis kinetic energy
    V = np.asarray(mints.ao_potential()) # AO basis nuclear potential
    ERI = np.asarray(mints.ao_eri())     # AO basis two-electron repulsion
    E_nuc = mol.p4mol.nuclear_repulsion_energy()

    nbs = S.shape[0] # Number of basis functions
    ndocc = int(mol.nel/2) # Number of doubly occupied orbitals

    mol.Hc = T + V # Core Hamiltonian

    A = mints.ao_overlap() # Psi4's Matrix power is more powerful than Numpy
    A.power(-0.5,1e-14)
    A = np.asarray(A) # Orthogonalization matrix S^(-1/2)

    eps, mol.C = diag(A,mol.Hc) # Coefficients and orbital energies from initial (guess) Fock 
    Cdocc = mol.C[:, :ndocc] # Coefficients of occupied orbitals only: only keep ndocc columns
    mol.D = np.einsum('ik,jk->ij',Cdocc,Cdocc) # Initial density

    ### START SCF ###
    E_old = 0
    D_old = np.zeros_like(mol.D)
    for scf_iter in range(1,max_iter):
        mol.J = np.einsum('rs,pqrs->pq',mol.D,ERI) # Compute Coulomb term
        mol.K = np.einsum('rs,prqs->pq',mol.D,ERI) # Compute Exhange term

        F = mol.Hc + 2*mol.J - mol.K # Compute Fock matrix

        E_SCF = np.einsum('ij,ij->',mol.D,mol.Hc+F) + E_nuc# Compute SCF energy

        grad = np.einsum('ij,jk,kl->il',F,mol.D,S) - np.einsum('ij,jk,kl->il',S,mol.D,F)
        grad = A.T @ grad @ A
        rms = np.mean(grad**2)**0.5

        if ((abs(E_SCF - E_old) < e_conv) and (rms < d_conv)):
            print("SCF converged in {} steps!\nSCF Energy = {}".format(scf_iter,E_SCF))
            mol.scf_computed = True
            mol.E_SCF = E_SCF
            mol.eps = eps
            break

        E_old = E_SCF

        D_old = mol.D

        eps, mol.C = diag(A,F)  
        Cdocc = mol.C[:, :ndocc] 
        mol.D = np.einsum('ik,jk->ij',Cdocc,Cdocc) # New density

