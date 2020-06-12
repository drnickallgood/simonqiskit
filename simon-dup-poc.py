import sys 
import logging
import matplotlib.pyplot as plt
import numpy as np
import operator
import itertools
#from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, IBMQ
from qiskit.providers.ibmq import least_busy
from collections import OrderedDict  

# AER is for simulators
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit import ClassicalRegister
from qiskit import QuantumRegister   
from qiskit import execute
from qiskit import IBMQ
#from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.visualization import circuit_drawer
from sympy import Matrix, pprint, MatrixSymbol, expand, mod_inverse
  
unitary_sim = Aer.get_backend('unitary_simulator')
n = 2
# Generate circuit
#qr = QuantumRegister(2*n,'q')
qr = QuantumRegister(2*n)
#cr = ClassicalRegister(n,'c')
cr = ClassicalRegister(n)
simonCircuit = QuantumCircuit(qr,cr)
uni_list = list()

def example():
    result2 = execute(simonCircuita, unitary_sim).result()
    unitary2 = result2.get_unitary(simonCircuita)
    if np.all((unitary2 == x) for x in uni_list):
        print("Duplicate")
    else:
        print("No duplicate")
        uni_list.append(unitary2)
     
simonCircuit.h(qr[0])
simonCircuit.h(qr[1])
simonCircuit.cx(qr[0],qr[2])
simonCircuit.x(qr[3])

result1 = execute(simonCircuit, unitary_sim).result()
unitary1 = result1.get_unitary(simonCircuit)
uni_list.append(unitary1)
print(len(uni_list))

# Generate circuit
#qra = QuantumRegister(2*n,'q')
qra = QuantumRegister(2*n)
#cra = ClassicalRegister(n,'c')
cra = ClassicalRegister(n)
simonCircuita = QuantumCircuit(qra,cra)

simonCircuita.h(qra[0])
simonCircuita.h(qra[1])

example()

print(len(uni_list))


