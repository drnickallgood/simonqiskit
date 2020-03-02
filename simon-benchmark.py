import sys
import matplotlib.pyplot as plt
import numpy as np
import operator
#from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, IBMQ

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

from qjob import QJob

def blackbox(period_string):
		#### Blackbox Function #####
		# QP's don't care about this, we do#
		#############################

		# Copy first register to second by using CNOT gates
		for i in range(n):
				#simonCircuit.cx(qr[i],qr[n+i])
				simonCircuit.cx(qr[i],qr[n+i])
				
		# get the small index j such it's "1"
		j = -1
		#reverse the string so that it takes
		s = period_string[::-1]
		for i, c in enumerate(s):
				if c == "1":
						j = i
						break
						
		# 1-1 and 2-1 mapping with jth qubit 
		# x is control to xor 2nd qubit with a
		for i, c in enumerate(s):
				if c == "1" and j >= 0:
						#simonCircuit.x(qr[j])
						simonCircuit.cx(qr[j], qr[n+i]) #the i-th qubit is flipped if s_i is 1
						#simonCircuit.x(qr[j])
						
						
		# Random peemutation
		# This part is how we can get by with 1 query of the oracle and better
		# simulates quantum behavior we'd expect
		perm = list(np.random.permutation(n))

		# init position
		init = list(range(n))

		i = 0

		while i < n:
				if init[i] != perm[i]:
						k = perm.index(init[i])
						simonCircuit.swap(qr[n+i],qr[n+k])	#swap gate on qubits
						init[i], init[k] = init[k], init[i] # mark the swapped qubits
				else:
						i += 1
						
		# Randomly flip qubit
		for i in range(n):
				if np.random.random() > 0.5:
						simonCircuit.x(qr[n+i])

		simonCircuit.barrier()
		### END OF BLACKBOX FUNCTION 
		
		return simonCircuit
		
def run_circuit(circuit,backend):
		# Default for this backend seems to be 1024 ibmqx2
		shots = 1024

		job = execute(simonCircuit,backend=backend, shots=shots)
		job_monitor(job,interval=2)
		results = job.result()
		return results

'''
		counts = results.get_counts()
		#print("Getting Results...\n")
		#print(qcounts)
		#print("")
		print("Submitting to IBM Q...\n")


		print("\nIBM Q Backend %s: Resulting Values and Probabilities" % qbackend)
		print("===============================================\n")
		print("Simulated Runs:",shots,"\n")

		# period, counts, prob,a0,a1,...,an
		#
		for key, val in qcounts.items():
				   prob = val / shots
				   print("Period:", key, ", Counts:", val, ", Probability:", prob)
				   
		print("")
'''

		
def guass_elim(results):
		# Classical post processing via Guassian elimination for the linear equations
		# Y a = 0
		# k[::-1], we reverse the order of the bitstring
		equations = list()
		lAnswer = [ (k[::-1],v) for k,v in results.get_counts().items() if k != "0"*n ]

		# Sort basis by probabilities
		lAnswer.sort(key = lambda x: x[1], reverse=True)

		Y = []
		for k, v in lAnswer:
						Y.append( [ int(c) for c in k ] )

		Y = Matrix(Y)

		Y_transformed = Y.rref(iszerofunc=lambda x: x % 2==0)

		# convert rational and negatives in rref
		def mod(x,modulus):
						numer,denom = x.as_numer_denom()
						return numer*mod_inverse(denom,modulus) % modulus

		# Deal with negative and fractial values
		Y_new = Y_transformed[0].applyfunc(lambda x: mod(x,2))

		#print("\nThe hidden period a0, ... a%d only satisfies these equations:" %(n-1))
		#print("===============================================================\n")
		rows,cols = Y_new.shape
		for r in range(rows):
						Yr = [ "a"+str(i)+"" for i,v in enumerate(list(Y_new[r,:])) if v==1]
						if len(Yr) > 0:
								tStr = " xor ".join(Yr)

								#single value is 0, only xor with perid string 0 to get
								if len(tStr) == 2:
										equations.append("period xor" + " 0 " + " = 0")
								else:
										equations.append("period" + " xor " + tStr + " = 0")

		return equations
										
def print_list(results):
# Sort list by value
		sorted_x = sorted(qcounts.items(), key=operator.itemgetter(1), reverse=True)
		print("Sorted list of result strings by counts")
		print("======================================\n")
		for i in sorted_x:
				print(i)
		#print(sorted_x)
		print("")


## function to get dot product of result string with the period string to verify, result should be 0
#check the wikipedia for simons formula 
# DOT PRODUCT IS MOD 2 !!!!
# Result XOR ?? = 0   -- this is what we're looking for!

# We have to verify the period string with the ouptput using mod_2 addition aka XOR
# Simply xor the period string with the output string	 
# Simply xor the period string with the output string, result must be 0 or 0b0
def verify_string(ostr, pstr):
	"""
	Verify string with period string
	Does dot product and then mod2 addition
	"""
	temp_list = list()
	# loop through outstring, make into numpy array
	for o in ostr:
		temp_list.append(int(o))
	
	ostr_arr = np.asarray(temp_list)
	temp_list.clear()
	
	# loop through period string, make into numpy array
   
	for p in pstr:
		temp_list.append(int(p))
		
	pstr_arr = np.asarray(temp_list)
	
	temp_list.clear()
	
	# Per Simosn, we do the dot product of the np arrays and then do mod 2
	results = np.dot(ostr_arr, pstr_arr)
	
	if results % 2 == 0:
		return True
	 
	return False

#### START ####			
# hidden stringsn
# We omit 00 as it's a trivial answer/solution

period_strings_5qubit = list()

#2-bit strings
s0 = "00"
s1 = "01"
s2 = "10"
s3 = "11"

#3-bits trings
s4 = "000"
s5 = "001"
s6 = "010"
s7 = "011"
s8 = "100"
s9 = "101"
s10 = "110"
s11 = "111"

#4-bit strings
s12 = "0000"
s13 = "0001"
s14 = "0010"
s15 = "0011"
s16 = "0100"
s17 = "0101"
s18 = "0110"
s19 = "0111"
s20 = "1000"
s21 = "1001"
s22 = "1010"
s23 = "1011"
s24 = "1100"
s25 = "1101"
s26 = "1110"
s27 = "1111"

# 5-bit strings
s28 = "10000"
s29 = "10001"
s30 = "10010"
s31 = "10011"
s32 = "10100"
s33 = "10101"
s34 = "10110"
s35 = "10111"
s36 = "11000"
s37 = "11001"
s38 = "11010"
s39 = "11011"
s40 = "11100"
s41 = "11101"
s42 = "11110"
s43 = "11111"

# 6-bit strings
s44 = "100000"
s45 = "100001"
s46 = "100010"
s47 = "100011"
s48 = "100100"
s49 = "100101"
s50 = "100110"
s51 = "100111"
s52 = "101000"
s53 = "101001"
s54 = "101010"
s55 = "101011"
s56 = "101100"
s57 = "101101"
s58 = "101110"
s59 = "101111"
s60 = "110000"
s61 = "110001"
s62 = "110010"
s63 = "110011"
s64 = "110100"
s65 = "110101"
s66 = "110110"
s67 = "110111"
s68 = "111000"
s69 = "111001"
s70 = "111010"
s71 = "111011"
s72 = "111100"
s73 = "111101"
s74 = "111110"
s75 = "111111"

# 7-bit strings
s76 = "1000000"
s77 = "1000001"
s78 = "1000010"
s79 = "1000011"
s80 = "1000100"
s89 = "1000101"
s90 = "1000110"
s91 = "1000111"
s92 = "1001000"
s93 = "1001001"
s94 = "1001010"
s95 = "1001011"
s96 = "1001100"
s97 = "1001101"
s98 = "1001110"
s99 = "1001111"
s100 = "1010000"
s101 = "1010001"
s102 = "1010010"
s103 = "1010011"
s104 = "1010100"
s105 = "1010101"
s106 = "1010110"
s106 = "1010111"
s107 = "1011000"
s108 = "1011001"
s109 = "1011010"
s110 = "1011011"
s111 = "1011100"
s112 = "1011101"
s113 = "1011110"
s114 = "1011111"
s115 = "1100000"
s116 = "1100001"
s117 = "1100011"
s118 = "1100100"
s119 = "1100101"
s120 = "1100110"
s121 = "1100111"
s122 = "1101000"
s123 = "1101001"
s124 = "1101010"
s125 = "1101011"
s126 = "1101100"
s127 = "1101101"
s128 = "1101110"
s129 = "1101111"
s130 = "1110000"
s131 = "1110001"
s132 = "1110010"
s133 = "1110011"
s134 = "1110100"
s135 = "1110101"
s136 = "1110110"
s137 = "1110111"
s138 = "1111000"
s138 = "1111001"
s139 = "1111010"
s140 = "1111011"
s141 = "1111100"
s142 = "1111101"
s143 = "1111110"
s144 = "1111111"



# 5-qubit strings
period_strings_5qubit.append(s0)
period_strings_5qubit.append(s1)
period_strings_5qubit.append(s2)
period_strings_5qubit.append(s3)

# 14-qubit strings, perhaps we can do 7 bit strings?
period_strings_14qubit = list()

period_strings_14qubit.append(s0)
period_strings_14qubit.append(s1)
period_strings_14qubit.append(s2)
period_strings_14qubit.append(s3)

period_strings_14qubit.append(s4)
period_strings_14qubit.append(s5)
period_strings_14qubit.append(s6)
period_strings_14qubit.append(s7)
period_strings_14qubit.append(s8)
period_strings_14qubit.append(s9)
period_strings_14qubit.append(s10)
period_strings_14qubit.append(s11)

period_strings_14qubit.append(s12)
period_strings_14qubit.append(s13)
period_strings_14qubit.append(s14)
period_strings_14qubit.append(s15)
period_strings_14qubit.append(s16)
period_strings_14qubit.append(s17)
period_strings_14qubit.append(s18)
period_strings_14qubit.append(s19)
period_strings_14qubit.append(s20)
period_strings_14qubit.append(s21)
period_strings_14qubit.append(s22)
period_strings_14qubit.append(s23)
period_strings_14qubit.append(s24)
period_strings_14qubit.append(s25)
period_strings_14qubit.append(s26)
period_strings_14qubit.append(s27)

period_strings_14qubit.append(s28)
period_strings_14qubit.append(s29)
period_strings_14qubit.append(s30)
period_strings_14qubit.append(s31)
period_strings_14qubit.append(s32)
period_strings_14qubit.append(s33)
period_strings_14qubit.append(s34)
period_strings_14qubit.append(s35)
period_strings_14qubit.append(s36)
period_strings_14qubit.append(s37)
period_strings_14qubit.append(s38)
period_strings_14qubit.append(s39)
period_strings_14qubit.append(s40)
period_strings_14qubit.append(s41)
period_strings_14qubit.append(s42)
period_strings_14qubit.append(s43)




# IBM Q stuff..
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')

# 14 qubit (broken?)
melbourne = provider.get_backend('ibmq_16_melbourne')

#5 qubit backends
ibmqx2 = provider.get_backend('ibmqx2')		# Yorktown
london = provider.get_backend('ibmq_london')
essex = provider.get_backend('ibmq_essex')
burlington = provider.get_backend('ibmq_burlington')
ourense = provider.get_backend('ibmq_ourense')
vigo = provider.get_backend('ibmq_vigo')

# 32 qubit qasm simulator
ibmq_sim = provider.get_backend('ibmq_qasm_simulator')

# Local Simulator, 
local_sim = Aer.get_backend('qasm_simulator')

circuitList = list()

backend_list = dict()
#backend_list['local_sim'] = local_sim

backend_list['ibmqx2'] = ibmqx2
#backend_list['london'] = london
#backend_list['essex'] = essex
#backend_list['burlington'] = burlington
#backend_list['ourense'] = ourense
#backend_list['melbourne'] = melbourne 
backend_list['vigo'] = vigo


# Make Circuits for all period strings!
for p in period_strings_5qubit:

		# Circuit name = Simon_+ period string
	#circuitName = "Simon-" + p

	circuitName = p
	n = len(p)
	# For simons, we use the first n registers for control qubits
	# We use the last n registers for data qubits.. which is why we need 2*n
	qr = QuantumRegister(2*n)
	cr = ClassicalRegister(n)
	simonCircuit = QuantumCircuit(qr,cr,name=circuitName)

	# Apply hadamards prior to oracle 
	for i in range(n):
		simonCircuit.h(qr[i])
		simonCircuit.barrier()

	#call oracle for period string
	simonCircuit = blackbox(p)

	# Apply hadamards after blackbox
	for i in range(n):
		simonCircuit.h(qr[i])

	simonCircuit.barrier()

	# Measure qubits, maybe change to just first qubit to measure
	simonCircuit.measure(qr[0:n],cr)
	
	circuitList.append(simonCircuit)


# Run loop to send circuits to IBMQ..
local_sim_ranJobs = list()
ibmqx2_ranJobs = list()
london_ranJobs = list()
essex_ranJobs = list()
burlington_ranJobs = list()
ourense_ranJobs = list()
vigo_ranJobs = list()
ibmq_sim_ranJobs = list()
melbourne_ranJobs = list()


print("\n===== SENDING DATA TO IBMQ BACKENDS... =====\n")	 
ranJobs = list() 

for name in backend_list:
	for circuit in circuitList:
		job = execute(circuit,backend=backend_list[name], shots=1024)
		print("Running job on backend: " + name)
		job_monitor(job,interval=5)
		# Custom object to hold the job, circuit, and backend
		qj = QJob(job,circuit,name)

		#print(qj.backend())
		# Append finished / ran job to list of jobs
		ranJobs.append(qj)


for qjob in ranJobs:
		# Results from each job
		results = qjob.job.result()

		# total counts from job
		counts = results.get_counts()

		# equations from each job
		equations = guass_elim(results)

		#period string encoded into name
		pstr = qjob.circuit.name

		#list of observed strings
		obs_strings = list()
	
		# Sorted strings from each job
		sorted_str = sorted(results.get_counts().items(), key=operator.itemgetter(1), reverse=True)

		# Get just the observed strings
		for string in sorted_str:
				obs_strings.append(string[0])

		# go through and verify strings
		for o in obs_strings:
				# Remember to re-reverse string so it's back to normal due to IBMQ Endianness
				if verify_string(o,pstr):
						#qjob.setCorrect()
						continue
				else:
						# lookup counts based on string
						if o == counts[0]:
							str_counts = counts[1]

						# Add value to incorrect holder in object
						qjob.setIncorrect(str_counts)

		# Now we haev the stats finished, let's store them in a list based on their backend name
		if qjob.backend() == "ibmqx2":
				ibmqx2_ranJobs.append(qjob)
		elif qjob.backend() == "london":
				london_ranJobs.append(qjob)
		elif qjob.backend() == "essex":
				burlington_ranJobs.append(qjob)
		elif qjob.backend() == "ourense":
				ourense_ranJobs.append(qjob)
		elif qjob.backend() == "vigo":
				vigo_ranJobs.append(qjob)
		elif qjob.backend() == "ibmq_sim":
				ibmq_sim_ranJobs.append(qjob)
		elif qjob.backend() == "melbourne":
				melbourne_ranJobs.append(qjob)
		elif qjob.backend() == "local_sim":
				local_sim_ranJobs.append(qjob)
		else:
				continue



backends_5qubit_ranJobs = dict()
backends_14qubit_ranJobs = dict()
backends_sims_ranJobs = dict()

#q5b = ["ibmqx2", "vigo", "ourense", "london", "essex", "burlington"]
q5b = ["ibmqx2", "vigo"]
q14b = ["melbourne"]
sims = ["local_sim"]
#sims = ["local_sim", "ibmq_SIM"]

backends_5qubit_ranJobs['ibmqx2'] = ibmqx2_ranJobs
backends_5qubit_ranJobs['vigo'] = vigo_ranJobs
backends_5qubit_ranJobs['ourense'] = ourense_ranJobs
backends_5qubit_ranJobs['london'] = london_ranJobs
backends_5qubit_ranJobs['essex'] = essex_ranJobs
backends_5qubit_ranJobs['burlington'] = burlington_ranJobs
backends_14qubit_ranJobs['melbourne'] = melbourne_ranJobs

backends_sims_ranJobs['local_sim'] = local_sim_ranJobs
#backends_sims['ibmq_sim'] = ibmq_sim_ranJobs

# The idea here is to loop through the dictionary by using a name in the list of names above
# as such then, we can call dictionaries in a loop with that name, which contain the list of
# ran jobs
def printStats(backend, job_list):
	'''
	backend: backend name
	job_list: list of ran jobs from backend

	'''

	total_correct = 0
	total_incorrect = 0

	# Total = shots = repeitions of circuit
	total = 1024
	pcorrect = 0.00
	pincorrect = 0.00

	for job in job_list:
		total_correct += job.getCorrect()
		total_incorrect += job.getIncorrect()
		total = total_correct + total_incorrect
		pcorrect = 100*(total_correct / total)
		if total_correct == total:
			pincorrect = 0.0000
		else:
			pincorrect = 100*(total_incorrect / total)


	print("\n===== RESULTS - " + backend + " =====\n")
	print("Total Results: " + str(total))
	print("Total Correct Results: " + str(total_correct) + " -- " + str(pcorrect) + "%") 
	print("Total Inorrect Results: " + str(total_incorrect) + " -- " + str(pincorrect) + "%")
	print("\n===================\n")


'''
print("\nIBMQX2 Results\n")
for q in ibmqx2_ranJobs:
	print("Correct: " + str(q.getCorrect()))
	print("Incorrect: " + str(q.getIncorrect()))

print("\nVIGO Results\n")
for q in vigo_ranJobs:
	print("Correct: " + str(q.getCorrect()))
	print("Incorrect: " + str(q.getIncorrect()))
'''


# for each backend name in the backend name list...
for backend in q5b:
	printStats(backend, backends_5qubit_ranJobs[backend])


# 14-qubit backend
'''
for backend in q14b:
	printStats(backend, backends_14qubit_ranJobs[backend])
'''






