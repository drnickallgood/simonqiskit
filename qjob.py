
class QJob:
    """    
    Enhanced quantum circuit to hold backend information and other potential useful reporting parameters
    """
        
    # Default total / shots = 1024 a circuit runs on a backend
    def __init__(self, job, circuit, backend_name,period,total=1024):
        self.job = job
        self.circuit = circuit
        self.backend_name = backend_name
        # of shots
        self.total = total
        self.incorrect = 0
        self.correct = 0
        self.period = period
        
    # Returns string of backend name
    def backend(self):
        return str(self.backend_name)
        
    # Returns quantumcircuit object
    def circuit(self):
        return self.circuit
    
    # Returns basejob object
    def job(self):
        return self.job

    def setCorrect(self, value):
        self.correct += value

    def setIncorrect(self, value):
        self.incorrect += value
        #self.correct = self.total - self.incorrect

    def getCorrect(self):
        return self.correct

    def getIncorrect(self):
        return self.incorrect

    def getPeriod(self):
        return self.period
        
 
        
        
