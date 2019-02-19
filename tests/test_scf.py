import numpy as np
import psi4
psi4.core.be_quiet()
import mesp

def test_scf():
    bas_list = ['STO-3G','DZ','DZP']
    mol_list = ['H2O', 'CO2', 'C2H4']
    for name in mol_list:
        geom = mesp.mollib[name]
        geom += '\nsymmetry c1'
        for bas in bas_list:
            psi4.core.clean()
            mol = mesp.Molecule(name,geom,bas)
            psi4.set_options({
                'basis':bas,
                'scf_type':'pk',
                'freeze_core':'false',
                'e_convergence':1e-12,
                'd_convergence':1e-12})
            psi4.set_module_options('SCF', 
                {'e_convergence': 1e-12, 'd_convergence': 1e-12,
                 'DIIS': True, 'scf_type':'pk'})

            print("Results for {}/SCF/{}".format(name,bas))
            mesp.scf.do_scf(mol)
            E_mesp = mol.E_SCF    
            E_psi4 = psi4.energy('SCF',molecule=mol.p4mol)
            
            print("mesp energy: {}".format(E_mesp))
            print("Psi4 energy: {}\n\n".format(E_psi4))
            psi4.compare_values(E_psi4,E_mesp,11,"RHF Energy")

if __name__=="__main__":
    test_scf()
