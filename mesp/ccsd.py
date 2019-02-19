import numpy as np
import psi4
import mesp

def do_ccsd(mol,
            e_conv = 1e-12,
            max_iter = 50,
            diis_start = 1,
            diis_max = 8,
            diis_step = 0):
    '''
    CCSD function
    
    Parameters
    ----------
    mol: MESP Molecule class
    e_conv: optional float (default = 1e-12), convergence criteria for energy
    max_iter: optional int (default = 50), maximum iterations for CCSD
    diis_start: optional int, first iteration where DIIS is performed
    diis_max: optional int, max number of Fock and gradient matrices held for DIIS extrapolation
    diis_step: optional int, allow `diis_step` relaxation cycles between DIIS extrapolation

    Notes
    ----------
    This is a spatial-orbital code derived using the Unitary Group Generator Approach.
    Equations were matched with Prof. Crawford's UGACC, factorized roughly according
    to the Stanton1991 CCSD intermediates.
    https://github.com/lothian/ugacc/blob/master/ccwfn.cc
    '''
    
    ### SETUP ###
    if not mol.scf_computed:
        mesp.do_scf(mol)

    # Make a Mints helper and grab integrals from psi4
    mints = psi4.core.MintsHelper(mol.p4wfn.basisset())
    C_p4 = psi4.core.Matrix.from_array(mol.C) # need p4 Matrix
    MO = np.asarray(mints.mo_eri(C_p4,C_p4,C_p4,C_p4)) # MO ERI
    MO = MO.swapaxes(1,2) # Physicist notation

    no = MO.shape[0] # convenient number of orbitals
    nocc = int(mol.nel / 2) # number of doubly occ orbitals
    nvir = no - nocc # number of vir orbitals

    F = mol.Hc + 2*mol.J - mol.K # AO-basis Fock
    F = np.einsum('uj,vi,uv', mol.C, mol.C, F) # MO-basis Fock

    # use slices to cut out occ and vir blocks
    o = slice(0,nocc) 
    v = slice(nocc,no)
    ov = (o,v) # tuple for passing into functions

    MO_ijab = MO[o,o,v,v] # <ij|ab>
    L = 2*MO - MO.swapaxes(2,3) # L_{ijab} = 2*<ij|ab> - <ij|ba>
                                # Also called gbar in Prof. Valeev's notation

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
        tau = build_tau(t1,t2)
        E_MP2_CORR = 2*np.einsum('ia,ia->',F[ov[0],ov[1]],t1)
        E_MP2_CORR += np.einsum('ijab,ijab->',tau,L[ov[0],ov[0],ov[1],ov[1]])
        E_MP2 = E_MP2_CORR + mol.E_SCF
        mol.E_MP2 = E_MP2
        print('MP2 energy (from CCSD) = {}'.format(E_MP2))

    # Start CCSD iterations
    E_old = 0.0
    t1_list = [t1.copy()]
    t2_list = [t2.copy()]
    t1_old = t1.copy()
    t2_old = t2.copy()
    r_list = [] # Residual list (DIIS error vectors)
    diis_count = 0
    for ccsd_iter in range(1,max_iter):
#        print("Iteration {} . . .".format(ccsd_iter))

        # Build tau intermediates
        tau = build_tau(t1,t2)
        tildetau = build_tildetau(t1,t2)

        # Build 2- and 4-index intermediates
        Fvv, Foo, Fov = build_F(F,L,ov,t1,t2,tildetau)
        Woooo, Wovov, Wovvo = build_W(MO,L,ov,t1,t2,tau)
        Z = build_Z(MO,ov,tau)

        # Build T1
        T1 = F[ov[0],ov[1]].copy()
        T1 += np.einsum('ie,ae->ia',t1,Fvv)
        T1 -= np.einsum('ma,mi->ia',t1,Foo)
        T1 += 2*np.einsum('miea,me->ia',t2,Fov)
        T1 -= np.einsum('miae,me->ia',t2,Fov)
        T1 += np.einsum('nf,nafi->ia',t1,L[ov[0],ov[1],ov[1],ov[0]])
        T1 += 2*np.einsum('inef,anef->ia',t2,MO[ov[1],ov[0],ov[1],ov[1]])
        T1 -= np.einsum('infe,anef->ia',t2,MO[ov[1],ov[0],ov[1],ov[1]])
        T1 -= np.einsum('mnea,mnei->ia',t2,L[ov[0],ov[0],ov[1],ov[0]])

        # Build T2
        T2 = MO[ov[0],ov[0],ov[1],ov[1]].copy()
        T2 += np.einsum('ijae,be->ijab',t2,Fvv)
        T2 += np.einsum('jibe,ae->ijab',t2,Fvv)
        T2 -= 0.5*np.einsum('ijae,mb,me->ijab',t2,t1,Fov)
        T2 -= 0.5*np.einsum('ijeb,ma,me->ijab',t2,t1,Fov)

        T2 -= np.einsum('imab,mj->ijab',t2,Foo)
        T2 -= np.einsum('mjab,mi->ijab',t2,Foo)
        T2 -= 0.5*np.einsum('imab,je,me->ijab',t2,t1,Fov)
        T2 -= 0.5*np.einsum('mjab,ie,me->ijab',t2,t1,Fov)

        T2 += np.einsum('mnab,mnij->ijab',tau,Woooo)
        T2 += np.einsum('ijef,abef->ijab',tau,MO[ov[1],ov[1],ov[1],ov[1]])
        T2 += np.einsum('ie,abej->ijab',t1,MO[ov[1],ov[1],ov[1],ov[0]])
        T2 += np.einsum('je,baei->ijab',t1,MO[ov[1],ov[1],ov[1],ov[0]])

        T2 -= np.einsum('ma,mbij->ijab',t1,MO[ov[0],ov[1],ov[0],ov[0]])
        T2 -= np.einsum('mb,maji->ijab',t1,MO[ov[0],ov[1],ov[0],ov[0]])
        T2 += np.einsum('imae,mbej->ijab',t2,Wovvo)
        T2 -= np.einsum('imea,mbej->ijab',t2,Wovvo)

        T2 += np.einsum('imae,mbej->ijab',t2,Wovvo)
        T2 += np.einsum('imae,mbje->ijab',t2,Wovov)
        T2 += np.einsum('mjae,mbie->ijab',t2,Wovov)
        T2 += np.einsum('imeb,maje->ijab',t2,Wovov)

        T2 += np.einsum('jmbe,maei->ijab',t2,Wovvo)
        T2 += np.einsum('jmbe,maie->ijab',t2,Wovov)
        T2 += np.einsum('jmbe,maei->ijab',t2,Wovvo)
        T2 -= np.einsum('jmeb,maei->ijab',t2,Wovvo)

        T2 -= np.einsum('ie,ma,mbej->ijab',t1,t1,MO[ov[0],ov[1],ov[1],ov[0]])
        T2 -= np.einsum('ie,mb,maje->ijab',t1,t1,MO[ov[0],ov[1],ov[0],ov[1]])
        T2 -= np.einsum('je,ma,mbie->ijab',t1,t1,MO[ov[0],ov[1],ov[0],ov[1]])
        T2 -= np.einsum('je,mb,maei->ijab',t1,t1,MO[ov[0],ov[1],ov[1],ov[0]])

        tmp = np.einsum('ma,mbij->ijab',t1,Z)
        T2 -= tmp
        T2 -= tmp.swapaxes(0,1).swapaxes(2,3)

        # Update t1 and t2 amplitudes
#        t1 += (T1 / D_ia)
#        t2 += (T2 / D_ijab)
        t1 = T1 / D_ia
        t2 = T2 / D_ijab

        # Calculate the current CCSD correlation energy
        tau = build_tau(t1,t2)
        one_E = 2*np.einsum('ia,ia->',F[ov[0],ov[1]],t1)
        two_E = np.einsum('ijab,ijab->',tau,L[ov[0],ov[0],ov[1],ov[1]])
        E_CCSD_CORR = one_E + two_E
#        print("one_E = {}\ntwo_E = {}".format(one_E,two_E))
#        print("CCSD Correlation Energy = {}\n\n".format(E_CCSD_CORR))

        # Check convergence
        if (abs(E_CCSD_CORR - E_old) < e_conv):
            E_CCSD = E_CCSD_CORR + mol.E_SCF
            mol.E_CCSD = E_CCSD
            mol.ccsd_computed = True
            print('CCSD converged in {} steps!\nCCSD Energy = {}'.format(ccsd_iter,E_CCSD))
            break
        else:
            E_old = E_CCSD_CORR
            t1_list.append(t1.copy())
            t2_list.append(t2.copy())
            r_t1 = (t1 - t1_old).ravel()
            r_t2 = (t2 - t2_old).ravel()
            r_list.append(np.concatenate((r_t1,r_t2)))
            if len(t1_list) > diis_max:
                del t1_list[0]
                del t2_list[0]
                del r_list[0]
            
            diis_count += 1
            if ccsd_iter >= diis_start and diis_count > diis_step: # See scf.py for DIIS notes
                B = np.empty((len(r_list)+1,len(r_list)+1))
                B[-1,:]  = -1
                B[:,-1]  = -1
                B[-1,-1] = 0
                for i in range(0,len(r_list)):
                    for j in range(0,len(r_list)):
                        if j > i: continue
                        B[i,j] = np.einsum('i,i->',r_list[i],r_list[j])
                        B[j,i] = B[i,j]
                B[:-1,:-1] /= np.abs(B[:-1,:-1]).max() 

                rhs = np.zeros(B.shape[0])
                rhs[-1] = -1

                c = np.linalg.solve(B,rhs)

                t1 = np.zeros_like(t1)
                t2 = np.zeros_like(t2)
                for i in range(len(r_list)):
                    t1 += c[i] * t1_list[i+1]
                    t2 += c[i] * t2_list[i+1]

                diis_count = 0

            t1_old = t1.copy() # Keep t1, whether it was DIIS-extrapolated or not
            t2_old = t2.copy()

    if mol.ccsd_computed == False:
        print("CCSD did not converge after {} steps.".format(max_iter))
        print("Current CCSD Correlation Energy = {}".format(E_CCSD_CORR))
        print("Current CCSD Energy = {}".format(E_CCSD_CORR + mol.E_SCF))

# TAU BUILD FNS
def build_tau(t1,t2):
    '''EQ 9'''
    tau = t2 + np.einsum('ia,jb->ijab',t1,t1)
    return tau

def build_tildetau(t1,t2):
    '''EQ 10'''
    tildetau = t2 + 0.5 * np.einsum('ia,jb->ijab',t1,t1)
    return tildetau

# 2-INDEX INTERMEDIATES
def build_Fvv(F,L,ov,t1,t2,tildetau):
    '''EQ 3'''
    fvv = F[ov[1],ov[1]].copy() # only need vir-vir block 
    fvv[np.diag_indices_from(fvv)] = 0 # only need off-diag elements
    fvv -= 0.5 * np.einsum('me,ma->ae',F[ov[0],ov[1]],t1)
    fvv += np.einsum('mf,mafe->ae',t1,L[ov[0],ov[1],ov[1],ov[1]])
    fvv -=  np.einsum('mnfa,mnfe->ae',tildetau,L[ov[0],ov[0],ov[1],ov[1]])
    return fvv

def build_Foo(F,L,ov,t1,t2,tildetau):
    '''EQ 4'''
    foo = F[ov[0],ov[0]].copy() # occ-occ block
    foo[np.diag_indices_from(foo)] = 0 # only need off-diag elements
    foo += 0.5 * np.einsum('me,ie->mi',F[ov[0],ov[1]],t1)
    foo += np.einsum('ne,mnie->mi',t1,L[ov[0],ov[0],ov[0],ov[1]])
    foo += np.einsum('infe,mnfe->mi',tildetau,L[ov[0],ov[0],ov[1],ov[1]])
    return foo

def build_Fov(F,L,ov,t1):
    '''EQ 5'''
    fov = F[ov[0],ov[1]].copy() + np.einsum('nf,mnef->me',t1,L[ov[0],ov[0],ov[1],ov[1]])
    return fov

def build_F(F,L,ov,t1,t2,tildetau):
    fvv = build_Fvv(F,L,ov,t1,t2,tildetau)
    foo = build_Foo(F,L,ov,t1,t2,tildetau)
    fov = build_Fov(F,L,ov,t1)
    return fvv, foo, fov

# 4-INDEX INTERMEDIATES
def build_Woooo(MO,ov,t1,t2,tau):
    '''EQ 6'''
    woooo = MO[ov[0],ov[0],ov[0],ov[0]].copy()
    woooo += np.einsum('je,mnie->mnij',t1,MO[ov[0],ov[0],ov[0],ov[1]])
    woooo += np.einsum('ie,mnej->mnij',t1,MO[ov[0],ov[0],ov[1],ov[0]])
    woooo += np.einsum('ijef,mnef->mnij',tau,MO[ov[0],ov[0],ov[1],ov[1]])
    return woooo

def build_Wovov(MO,ov,t1,t2,tau):
    '''EQ 7- now Wovov and Z'''
    wovov = -1*MO[ov[0],ov[1],ov[0],ov[1]].copy()
    wovov -= np.einsum('jf,mbfe->mbje',t1,MO[ov[0],ov[1],ov[1],ov[1]])
    wovov += np.einsum('nb,mnje->mbje',t1,MO[ov[0],ov[0],ov[0],ov[1]])
    tmp = np.einsum('jf,nb->jnfb',t1,t1)
    wovov += np.einsum('mnfe,jnfb->mbje',MO[ov[0],ov[0],ov[1],ov[1]],(0.5*t2+tmp))
    return wovov

def build_Wovvo(MO,L,ov,t1,t2):
    '''EQ 8'''
    wovvo = MO[ov[0],ov[1],ov[1],ov[0]].copy()
    wovvo += np.einsum('jf,mbef->mbej',t1,MO[ov[0],ov[1],ov[1],ov[1]])
    wovvo -= np.einsum('nb,mnej->mbej',t1,MO[ov[0],ov[0],ov[1],ov[0]])
    tmp = np.einsum('jf,nb->jnfb',t1,t1)
    wovvo -= np.einsum('mnef,jnfb->mbej',MO[ov[0],ov[0],ov[1],ov[1]],(0.5*t2+tmp))
    wovvo += 0.5*np.einsum('mnef,njfb->mbej',L[ov[0],ov[0],ov[1],ov[1]],t2)
    return wovvo

def build_W(MO,L,ov,t1,t2,tau):
    woooo = build_Woooo(MO,ov,t1,t2,tau)
    wovov = build_Wovov(MO,ov,t1,t2,tau)
    wovvo = build_Wovvo(MO,L,ov,t1,t2)
    return woooo,wovov,wovvo

def build_Z(MO,ov,tau):
    '''Necessary to avoid explicit building of Wvvvv'''
    z_ovoo = np.einsum('mbef,ijef->mbij',MO[ov[0],ov[1],ov[1],ov[1]].copy(),tau)
    return z_ovoo
