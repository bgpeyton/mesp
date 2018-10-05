import psi4

class Molecule:
    def __init__(self,name,geom,bas):
        # Initialize passed variables
        self.name = name
        self.geom = geom

        # Build a Psi4 molecule to pull basis, number of electrons, etc
        self.p4mol = psi4.geometry(geom)
        self.p4mol.update_geometry()
        self.p4wfn = psi4.core.Wavefunction.build(self.p4mol,bas)
        self.nel = self.p4wfn.nalpha() * 2

        # Leave some space to store other things
        self.C  = None  # MO coefficient matrix
        self.D  = None  # Density
        self.g  = None  # bleh
        self.Hc = None  # Core Hamiltonian
        self.H  = None  # Hamiltonian
        self.J  = None  # Coulomb
        self.K  = None  # Exchange

        # Progress
        self.scf_computed = False
        self.E_SCF = False
