
class QJob:
    """	
    Enhanced quantum circuit to hold backend information and other potential useful reporting parameters
    """
        
    def __init__(self, job, circuit, backend_name,correct=0,incorrect=0):
        self.job = job
        self.circuit = circuit
        self.backend_name = backend_name
        self.correct = correct
        self.incorrect = incorrect
        
    # Returns string of backend name
    def backend(self):
        return str(self.backend_name)
        
    # Returns quantumcircuit object
    def circuit(self):
        return self.circuit
    
    # Returns basejob object
    def job(self):
        return self.job

    def setCorrect(self):
        self.correct += 1

    def setIncorrect(self):
        self.incorrect += 1

    def correct(self):
        return self.correct

    def incorrect(self):
        return self.incorrect
 
        
		
