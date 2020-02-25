
class QJob:
    """	
    Enhanced quantum circuit to hold backend information and other potential useful reporting parameters
    """
        
    # Correct / Incorrect 
    _correct = 0
    _incorrect = 0
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

    def setCorrect(self):
        self._correct += 1

    def setIncorrect(self):
        self._incorrect += 1

    def correct(self):
        return str(self._correct)

    def incorrect(self):
        return str(self._incorrect)
 
        
		
