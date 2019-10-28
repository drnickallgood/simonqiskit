from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.providers.aer import noise
import pprint

# Choose a real device to simulate
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
device = provider.get_backend('ibmq_vigo')
properties = device.properties()
coupling_map = device.configuration().coupling_map


# Generate an Aer noise model for device
noise_model = noise.device.basic_device_noise_model(properties)
basis_gates = noise_model.basis_gates

test2 = noise_model.to_dict()
print("Noise Model")
pprint.pprint(test2)
print("\nCoupling Map")
print(coupling_map)
print("\nProperties")
test = properties.to_dict()

pprint.pprint(test)

#print(properties)
# Generate a quantum circuit
#qc = QuantumCircuit(2, 2)

#qc.h(0)
#qc.cx(0, 1)
#qc.measure([0, 1], [0, 1])

# Perform noisy simulation
#backend = Aer.get_backend('qasm_simulator')
'''
job_sim = execute(qc, backend,
                  coupling_map=coupling_map,
                  noise_model=noise_model,
                  basis_gates=basis_gates)
'''
#sim_result = job_sim.result()
#print(sim_result.get_counts(qc))
              
