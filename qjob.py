""" 
Enhanced quantum circuit to hold backend information and other potential useful reporting parameters

"""

class QJob:

    def __init__(self, job, circuit, backend):
        self.job = job
        self.circuit = circuit
        self.backend = backend
        
    # Returns string of backend name
    def backend(self):
        return backend
        
    # Returns quantumcircuit object
    def circuit(self):
        return circuit
    
    # Returns basejob object
    def job(self):
        return job
        
 
        
		