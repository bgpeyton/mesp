import numpy as np
import psi4
import mesp

def do_so_ccsd(mol,
            e_conv = 1e-12,
            max_iter = 50,
            save_t = False):
    '''
    Spin Orbital CCSD function
    
    Parameters
    ----------
    mol: MESP Molecule class
    max_iter: int, maximum iterations for CCSD
    save_t: optional bool (default = False), save final t-amplitudes
    '''
    
    ### SETUP ###
    if not mol.scf_computed:
        mesp.do_scf(mol)

    # Make a Mints helper and grab integrals from psi4
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
    F = np.kron(F,I) # Expand and spin-tegrate, thanks Vibin!

    # use slices to cut out occ and vir blocks
    o = slice(0,nocc) 
    v = slice(nocc,nso)
    ov = (o,v) # tuple for passing into functions

    MO_ijab = MO[o,o,v,v] # <ij||ab>

    # Grab orbital energies. I know I could just tile mol.eps,
    # but this is a good check if my Fock is correct
    diag = np.diag(F) # orbital eigenvalues in a 1D array
    F_occ = diag[:nocc] # grab first nocc diag elements
    F_vir = diag[nocc:] # grab the last nvir elements

    # Clever construction of denominator matrices- thanks Ashutosh!
    D_ia = F_occ.reshape(-1,1) - F_vir # EQ 12
    D_ijab = F_occ.reshape(-1,1,1,1) + F_occ.reshape(-1,1,1) - F_vir.reshape(-1,1) - F_vir # EQ 13

    # Initial guesses
    t1 = np.zeros((nocc,nvir))
    t2 = MO_ijab / D_ijab

    if not mol.mp2_computed:
        E_MP2_CORR = 0.25 * np.einsum('ijab,ijab->',MO_ijab,t2)
        E_MP2 = E_MP2_CORR + mol.E_SCF
        mol.E_MP2 = E_MP2
        print('MP2 energy (from CCSD) = {}'.format(E_MP2))

    # Start CCSD iterations
    E_old = 0.0
    for ccsd_iter in range(1,max_iter):
        # Build 2- and 4-index intermediates
        Fae = build_Fae(F,MO,ov,t1,t2)
        Fmi = build_Fmi(F,MO,ov,t1,t2)
        Fme = build_Fme(F,MO,ov,t1)
        Wmnij = build_Wmnij(MO,ov,t1,t2)
        Wabef = build_Wabef(MO,ov,t1,t2)
        Wmbej = build_Wmbej(MO,ov,t1,t2)

        # Build T1
        T1 = F[ov[0],ov[1]].copy()
        T1 += np.einsum('ie,ae->ia',t1,Fae)
        T1 -= np.einsum('ma,mi->ia',t1,Fmi)
        T1 += np.einsum('imae,me->ia',t2,Fme)
        T1 -= np.einsum('nf,naif->ia',t1,MO[ov[0],ov[1],ov[0],ov[1]])
        T1 -= 0.5 * np.einsum('imef,maef->ia',t2,MO[ov[0],ov[1],ov[1],ov[1]])
        T1 -= 0.5 * np.einsum('mnae,nmei->ia',t2,MO[ov[0],ov[0],ov[1],ov[0]])

        # Build T2
        T2 = MO[ov[0],ov[0],ov[1],ov[1]].copy()
        tmp = 0.5 * np.einsum('mb,me->be',t1,Fme)
        tmp = Fae - tmp
        T2 += np.einsum('ijae,be->ijab',t2,tmp)
        T2 -= np.einsum('ijbe,ae->ijab',t2,tmp) # Questionable- rebuild tmp with P(ab)?

        tmp = 0.5 * np.einsum('je,me->mj',t1,Fme)
        tmp = Fmi + tmp
        T2 -= np.einsum('imab,mj->ijab',t2,tmp)
        T2 += np.einsum('jmab,mi->ijab',t2,tmp) # Also questionable

        tau = build_tau(t1,t2)
        T2 += 0.5 * np.einsum('mnab,mnij->ijab',tau,Wmnij)
        T2 += 0.5 * np.einsum('ijef,abef->ijab',tau,Wabef)

        T2 -= np.einsum('ie,ma,mbej->ijab',t1,t1,MO[ov[0],ov[1],ov[1],ov[0]])
        T2 += np.einsum('imae,mbej->ijab',t2,Wmbej)
        T2 += np.einsum('ie,mb,maej->ijab',t1,t1,MO[ov[0],ov[1],ov[1],ov[0]])
        T2 -= np.einsum('imbe,maej->ijab',t2,Wmbej)
        T2 += np.einsum('je,ma,mbei->ijab',t1,t1,MO[ov[0],ov[1],ov[1],ov[0]])
        T2 -= np.einsum('jmae,mbei->ijab',t2,Wmbej)
        T2 -= np.einsum('je,mb,maei->ijab',t1,t1,MO[ov[0],ov[1],ov[1],ov[0]])
        T2 += np.einsum('jmbe,maei->ijab',t2,Wmbej)

        T2 += np.einsum('ie,abej->ijab',t1,MO[ov[1],ov[1],ov[1],ov[0]])
        T2 -= np.einsum('je,abei->ijab',t1,MO[ov[1],ov[1],ov[1],ov[0]])

        T2 -= np.einsum('ma,mbij->ijab',t1,MO[ov[0],ov[1],ov[0],ov[0]])
        T2 += np.einsum('mb,maij->ijab',t1,MO[ov[0],ov[1],ov[0],ov[0]])

        # Update t1 and t2 amplitudes
        t1 = T1 / D_ia
        t2 = T2 / D_ijab

        # Calculate the current CCSD correlation energy
        E_CCSD_CORR = np.einsum('ia,ia->',F[ov[0],ov[1]],t1)
        E_CCSD_CORR += 0.25 * np.einsum('ijab,ijab->',MO_ijab,t2)
        E_CCSD_CORR += 0.5 * np.einsum('ijab,ia,jb->',MO_ijab,t1,t1)
#        print("Iteration {} . . .".format(ccsd_iter))
#        print("CCSD Correlation Energy = {}".format(E_CCSD_CORR))

        # Check convergence
        if (abs(E_CCSD_CORR - E_old) < e_conv):
            E_CCSD = E_CCSD_CORR + mol.E_SCF
            mol.E_CCSD = E_CCSD
            mol.ccsd_computed = True
            print("CCSD converged in {} steps!\nCCSD Energy = {}".format(ccsd_iter,E_CCSD))
            if save_t:
                mol.t1 = t1
                mol.t2 = t2
            break
        else:
            E_old = E_CCSD_CORR
    if mol.ccsd_computed == False:
        print("CCSD did not converge after {} steps.".format(max_iter))
        print("Current CCSD Correlation Energy = {}".format(E_CCSD_CORR))

# TAU BUILD FNS
def build_tau(t1,t2):
    '''EQ 9'''
    tau = t2 + np.einsum('ia,jb->ijab',t1,t1)
    tau -= np.einsum('ib,ja->ijab',t1,t1)
    return tau

def build_tildetau(t1,t2):
    '''EQ 10'''
    tildetau = t2 + 0.5 * np.einsum('ia,jb->ijab',t1,t1)
    tildetau -= 0.5 * np.einsum('ib,ja->ijab',t1,t1)
    return tildetau

# 2-INDEX INTERMEDIATES
def build_Fae(F,MO,ov,t1,t2):
    '''EQ 3'''
    fae = F[ov[1],ov[1]].copy() # only need vir-vir block 
    fae[np.diag_indices_from(fae)] = 0 # only need off-diag elements
    fae -= 0.5 * np.einsum('me,ma->ae',F[ov[0],ov[1]],t1)
    fae += np.einsum('mf,mafe->ae',t1,MO[ov[0],ov[1],ov[1],ov[1]])
    tildetau = build_tildetau(t1,t2)
    fae -= 0.5 * np.einsum('mnaf,mnef->ae',tildetau,MO[ov[0],ov[0],ov[1],ov[1]])
    return fae

def build_Fmi(F,MO,ov,t1,t2):
    '''EQ 4'''
    fmi = F[ov[0],ov[0]].copy() # occ-occ block
    fmi[np.diag_indices_from(fmi)] = 0 # only need off-diag elements
    fmi += 0.5 * np.einsum('ie,me->mi',t1,F[ov[0],ov[1]])
    fmi += np.einsum('ne,mnie->mi',t1,MO[ov[0],ov[0],ov[0],ov[1]])
    tildetau = build_tildetau(t1,t2)
    fmi += 0.5 * np.einsum('inef,mnef->mi',tildetau,MO[ov[0],ov[0],ov[1],ov[1]])
    return fmi

def build_Fme(F,MO,ov,t1):
    '''EQ 5'''
    fme = F[ov[0],ov[1]].copy() + np.einsum('nf,mnef->me',t1,MO[ov[0],ov[0],ov[1],ov[1]])
    return fme

# 4-INDEX INTERMEDIATES
def build_Wmnij(MO,ov,t1,t2):
    '''EQ 6'''
    wmnij = MO[ov[0],ov[0],ov[0],ov[0]].copy()
    wmnij += np.einsum('je,mnie->mnij',t1,MO[ov[0],ov[0],ov[0],ov[1]])
    wmnij -= np.einsum('ie,mnje->mnij',t1,MO[ov[0],ov[0],ov[0],ov[1]])
    tau = build_tau(t1,t2)
    wmnij += 0.25 * np.einsum('ijef,mnef->mnij',tau,MO[ov[0],ov[0],ov[1],ov[1]])
    return wmnij

def build_Wabef(MO,ov,t1,t2):
    '''EQ 7'''
    wabef = MO[ov[1],ov[1],ov[1],ov[1]].copy()
    wabef -= np.einsum('mb,amef->abef',t1,MO[ov[1],ov[0],ov[1],ov[1]])
    wabef += np.einsum('ma,bmef->abef',t1,MO[ov[1],ov[0],ov[1],ov[1]])
    tau = build_tau(t1,t2)
    wabef += 0.25 * np.einsum('mnab,mnef->abef',tau,MO[ov[0],ov[0],ov[1],ov[1]])
    return wabef

def build_Wmbej(MO,ov,t1,t2):
    '''EQ 8'''
    wmbej = MO[ov[0],ov[1],ov[1],ov[0]].copy()
    wmbej += np.einsum('jf,mbef->mbej',t1,MO[ov[0],ov[1],ov[1],ov[1]])
    wmbej -= np.einsum('nb,mnej->mbej',t1,MO[ov[0],ov[0],ov[1],ov[0]])
    tmp = np.einsum('jf,nb->jnfb',t1,t1)
    wmbej -= np.einsum('jnfb,mnef->mbej',0.5*t2+tmp,MO[ov[0],ov[0],ov[1],ov[1]])
    return wmbej
