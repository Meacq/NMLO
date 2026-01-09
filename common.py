import numpy as np
from collections import defaultdict
import heapq, os
from datetime import datetime
from cqlib.utils import QasmToQcis 
from qiskit.converters import circuit_to_dag
import rustworkx as rx
from cqlib.utils import QcisToQasm, QasmToQcis
from qiskit import QuantumCircuit as qiskit_QuantumCircuit

# custom sort key function
def sort_key(item):
    # extract the number part from the name field
    number = int(''.join(filter(str.isdigit, os.path.basename(item))))
    return number

def cal_time_diff(begin_time,end_time):
    # calculate the time difference
    time_difference = end_time - begin_time
    # convert the time difference to seconds
    seconds = float(time_difference.total_seconds())
    return seconds

def merge_dicts(dicts):
    merged_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = value
            else:
                merged_dict[key] += value
    return merged_dict

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b.conj())
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def get_top_n_indices_with_duplicates(lst, n):
    if n <= 0:
        return []
    
    # record the indices of each value
    value_to_indices = defaultdict(list)
    for idx, val in enumerate(lst):
        value_to_indices[val].append(idx)
    
    # get the top N maximum values
    top_n_values = heapq.nlargest(n, set(lst))

    # collect all indices of these maximum values
    indices = []
    for value in top_n_values:
        indices.extend(value_to_indices[value])
        if len(indices) >= n:
            break
    
    # if the number of collected indices exceeds n, only keep the first n indices
    return indices[:n]

def get_execute_config(machine_time, repeat, shots, layout, layout_id, expected_fidelity, real_fidelity, begin_time, end_time):
    config = {}
    config['machine_time'] = machine_time
    config['repeat'] = repeat
    config['shots'] = shots
    config['layout'] = layout
    config['layout_id'] = layout_id
    config['expected_fidelity'] = expected_fidelity
    config['real_fidelity'] = real_fidelity
    config['begin_time'] = begin_time.strftime("%Y-%m-%d %H:%M:%S")
    config['end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    config['cost_time'] = f"{cal_time_diff(begin_time,end_time)} s"
    return config
    
def remove_measure(circ):
    # create a new quantum circuit without measure gates
    # copy the quantum circuit without classical bits
    new_qc = qiskit_QuantumCircuit(circ.num_qubits)

    # traverse and add non-measurement gates to the new circuit
    for op, qargs, cargs in circ.data:
        if not isinstance(op, Measure):
            new_qc.append(op, qargs) 
    return new_qc

def get_measured_qubits(qc):
    """
    # get all qubits that have measurement gates and sort them in descending order
    
    # parameter:
    qc (QuantumCircuit)
    
    # return:
    list: qubits that have measurement gates sorted in descending order
    """
    measured_qubits = set()
    for instruction in qc.data:
        if instruction.operation.name == 'measure':
            measured_qubits.add(instruction.qubits[0].index)
    
    # sort the qubits in descending order
    return sorted(measured_qubits, reverse=True)

def get_idle_qubits(qc):
    """
    # get all qubits that have no operations and sort them in descending order

    # parameter:
    qc (QuantumCircuit)

    # return:
    list: qubits that have no operations sorted in descending order
    """
    total_qubits = qc.num_qubits
    all_qubits = set(range(total_qubits))
    
    active_qubits = set()
    for instruction in qc.data:
        for qubit in instruction.qubits:
            active_qubits.add(qubit.index)
    
    idle_qubits = all_qubits - active_qubits
    
    # sort the qubits in descending order
    return sorted(idle_qubits, reverse=True)

def generate_binary_strings_with_fixed_bits(measured_qubits, fixed_qubits, fixed_value, total_qubits):
    """
    create all possible binary strings based on the measured qubits and fixed qubits,
    and sort them in descending order of [4, 3, 2, 1, 0].
    
    # parameter:
    measured_qubits (list): qubits that have measurement gates sorted in descending order
    fixed_qubits (list): qubits that have fixed values sorted in descending order
    fixed_value (int): fixed value of the fixed qubits (0 or 1)
    total_qubits (int): total number of qubits, default is 5

    # return:
    list: all possible binary strings sorted in descending order of [4, 3, 2, 1, 0]
    """
    binary_strings = []
    
    # generate all possible binary combinations for the measured qubits
    num_measured_qubits = len(measured_qubits)
    for i in range(2 ** num_measured_qubits):
        binary_string = format(i, f'0{num_measured_qubits}b')
        
        # create a new string with total_qubits length and insert fixed bits
        full_binary_string = ['0'] * total_qubits
        
        # insert the measured qubit values into the corresponding positions
        for j, qubit in enumerate(measured_qubits):
            full_binary_string[qubit] = binary_string[j]

        # insert the fixed bits
        for qubit in fixed_qubits:
            full_binary_string[qubit] = str(fixed_value)
        
        # ressort the binary string in descending order of [4, 3, 2, 1, 0]
        ordered_binary_string = ''.join([full_binary_string[i] for i in [total_qubits-1-x for x in range(total_qubits)]])

        # add the ordered binary string to the result list
        binary_strings.append(ordered_binary_string)
    
    return binary_strings

def post_result(result, small_qc):
    """
    # post process the experiment results, based on the measured qubits and idle qubits,
    # generate all possible binary strings, and sort the results according to these strings.
    
    # parameter:
    result (np.array): experiment results array, shape is (2^n,), n is the number of measured qubits
    small_qc (QuantumCircuit): quantum circuit, containing measurement gates and idle qubits
    
    return:
    np.array: sorted experiment results array, shape is (2^n,)
    """
    measured_qubits = get_measured_qubits(small_qc)
    idle_qubits = get_idle_qubits(small_qc)
    total_qubits = small_qc.num_qubits
    binary_strings = generate_binary_strings_with_fixed_bits(measured_qubits, idle_qubits, 0, total_qubits)
    indices = [int(i,2) for i in binary_strings]
    new_arr = result[indices]
    return new_arr  

def validate_connectivity(qcis_str):
    """
    # validate the connectivity of the quantum circuit, check if it is fully connected.
    
    # parameter:
    qcis_str (str): Qcis format quantum circuit string
    
    return:
    bool: True if the circuit is fully connected, False otherwise
    """
    qasm_str = QcisToQasm.convert_qcis_to_qasm(qcis_str)
    _qc = qiskit_QuantumCircuit.from_qasm_str(qasm_str)
    dag = circuit_to_dag(_qc)
    qubits = dag.qubits
    qubit_indices = {qubit: index for index, qubit in enumerate(qubits)}

    interactions = []
    graph_nodes = []
    for node in dag.op_nodes(include_directives=False):
        len_args = len(node.qargs)
        if len_args == 2:
            interactions.append((qubit_indices[node.qargs[0]], qubit_indices[node.qargs[1]]))
            graph_nodes.append(qubit_indices[node.qargs[0]])
            graph_nodes.append(qubit_indices[node.qargs[1]])
        elif len_args == 1:
            graph_nodes.append(qubit_indices[node.qargs[0]])

    im_graph = rx.PyGraph(multigraph=False)
    im_graph.add_nodes_from(range(_qc.num_qubits))
    unuesd_nodes = list(set([x for x in range(max(qubit_indices.values())+1)]) - set(graph_nodes))
    im_graph.remove_nodes_from(unuesd_nodes)
    im_graph.add_edges_from_no_data(interactions)
    return rx.is_connected(im_graph)

class PCOLOR():

    def __init__(self):
        self.RED_BOLD = "\033[1;31m"      # red bold
        self.GREEN_BOLD = "\033[1;32m"    # green bold
        self.YELLOW_BOLD = "\033[1;33m"   # yellow bold
        self.BLUE_BOLD = "\033[1;34m"     # blue bold
        self.MAGENTA_BOLD = "\033[1;35m"  # magenta bold
        self.CYAN_BOLD = "\033[1;36m"     # cyan bold
        self.WHITE_BOLD = "\033[1;37m"    # white bold
        self.BLACK_BOLD = "\033[1;30m"    # black bold
        self.RESET = "\033[0m"            # reset style
