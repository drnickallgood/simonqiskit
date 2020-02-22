
class QJob:
	"""
	Enhanced quantum circuit to hold backend information and other potential useful reporting parameters
	"""

	#Correct  and incorrect values
	_correct = 0
	_incorrect = 0

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
        pass

    def setIncorrect(self):
        pass

    def correct(self):
        return str(self._correct)

    def incorrect(self):
        return str(self._incorrect)
 
        
		
