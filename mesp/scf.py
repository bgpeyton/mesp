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
        max_iter = 50,
        diis_start = 3,
        diis_max = 6,
        diis_step = 0):
    '''
    SCF function
    
    Parameters
    ----------
    mol: MESP Molecule class
    e_conv: float
    d_conv: float
    max_iter: int, maximum iterations for SCF
    diis_start: int, first iteration where DIIS is performed
    diis_max: int, max number of Fock and gradient matrices held for DIIS extrapolation
    diis_step: int, allow `diis_step` full normal scf cycles between DIIS extrapolation
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

    F_list = [] # List of F for DIIS
    r_list = [] # List of residuals (gradients) for DIIS
    diis_count = 0 # Keep track of steps between DIIS

    ### START SCF ###
    E_old = 0
    D_old = np.zeros_like(mol.D)
    for scf_iter in range(1,max_iter+1):
        mol.J = np.einsum('rs,pqrs->pq',mol.D,ERI) # Compute Coulomb term
        mol.K = np.einsum('rs,prqs->pq',mol.D,ERI) # Compute Exhange term

        F = mol.Hc + 2*mol.J - mol.K # Compute Fock matrix

        E_SCF = np.einsum('ij,ij->',mol.D,mol.Hc+F) + E_nuc # Compute SCF energy

        grad = np.einsum('ij,jk,kl->il',F,mol.D,S) - np.einsum('ij,jk,kl->il',S,mol.D,F)
        grad = A.T @ grad @ A # Compute orthogonormalized orbital gradient
        rms = np.mean(grad**2)**0.5 # Compute RMSD of the orbital gradient
        
        F_list.append(F)
        r_list.append(grad)
        if len(F_list) > diis_max:
            del F_list[0]
            del r_list[0]

        if debug:
            print("Iter {}: E = {}, rms = {}".format(scf_iter,E_SCF,rms))

        if ((abs(E_SCF - E_old) < e_conv) and (rms < d_conv)):
            print("SCF converged in {} steps!\nSCF Energy = {}".format(scf_iter,E_SCF))
            mol.scf_computed = True
            mol.E_SCF = E_SCF
            mol.eps = eps
            break
    
        diis_count += 1
        if scf_iter >= diis_start and diis_count > diis_step:
            B = np.empty((len(F_list)+1,len(F_list)+1)) # Build B matrix to solve Pulay Eqn
            B[-1,:]  = -1 #   [<r|r> ..  -1] [c1]   [0 ]
            B[:,-1]  = -1 #   [ ..   ..  ..] [..] = [..]
            B[-1,-1] = 0  #   [ -1   ..   0] [L ]   [-1]
            for i in range(0,len(F_list)): # Compute overlaps <r_i|r_j>
                for j in range(0,len(F_list)):
                    if j > i: continue
                    B[i,j] = np.einsum('ij,ij->',r_list[i],r_list[j])
                    B[j,i] = B[i,j] # B is symmetric!
            B[:-1,:-1] /= np.abs(B[:-1,:-1]).max() # Normalize

            rhs = np.zeros(B.shape[0])
            rhs[-1] = -1

            c = np.linalg.solve(B,rhs) # Solve Pulay eq for coefficient vector

            F = np.zeros_like(F) # Solve DIIS Fock
            for i in range(c.shape[0] - 1):
                F += c[i] * F_list[i] # Linear combination of F_list

            diis_count = 0 # Reset DIIS counter

        E_old = E_SCF # Get ready for the next SCF cycle
        D_old = mol.D

        eps, mol.C = diag(A,F)  
        Cdocc = mol.C[:, :ndocc] 
        mol.D = np.einsum('ik,jk->ij',Cdocc,Cdocc) # New density
    
    if mol.scf_computed == False:
        print("SCF did not converge.")
