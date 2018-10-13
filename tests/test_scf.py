import numpy as np
import psi4
psi4.core.be_quiet()
import mesp

#geom = """
#    O
#    H 1 1
#    H 1 1 2 104.5
#    symmetry c1
#"""
#bas = "sto-3g"
#mol = mesp.Molecule('H2O',geom,bas)

def test_scf():
    bas_list = ['STO-3G','DZ','DZP']
    for name,geom in mesp.mollib.items():
        for bas in bas_list:
            psi4.core.clean()
            geom += '\nsymmetry c1'
            mol = mesp.Molecule(name,geom,bas)
            psi4.set_options({
                'basis':bas,
                'scf_type':'pk',
                'freeze_core':'false',
                'e_convergence':1e-8,
                'd_convergence':1e-8})
            psi4.set_module_options('SCF', 
                    {'DIIS': False,
                     'MAXITER': 80})

            print("Results for {}/SCF/{}".format(name,bas))
            if name == 'C2H2':
                mesp.scf.do_scf(mol,max_iter=100,debug=True)
            else:
                mesp.scf.do_scf(mol)
            E_mesp = mol.E_SCF    
            E_psi4 = psi4.energy('SCF',molecule=mol.p4mol)
            
            print("mesp energy: {}".format(E_mesp))
            print("Psi4 energy: {}\n\n".format(E_psi4))
            assert np.allclose(E_psi4,E_mesp)

if __name__=="__main__":
    test_scf()
