
class QJob:
    """	
    Enhanced quantum circuit to hold backend information and other potential useful reporting parameters
    """
        
    def __init__(self, job, circuit, backend,correct=None,incorrect=None):
        self.job = job
        self.circuit = circuit
        self.backend = backend
        self.correct = correct
        self.incorrect = incorrect
        
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
        self.correct += 1

    def setIncorrect(self):
        self.incorrect += 1

    def correct(self):
        return self.correct

    def incorrect(self):
        return self.incorrect
 
        
		
