from cqlib import Circuit
import numpy as np
import math, re, json
from tianyan_utils import replace_pattern, tianyan_machine_config, default_cost
from common import validate_connectivity
import pandas as pd

# define quantum gate execution time, unit(ns)
gate_execution_times = {
    "I": 60,
    "X2M": 60,
    "Y2M": 60,
    "B": 60,
    "XY2P": 60,
    "XY2M": 60,
    "Y2P": 60,
    "X2P": 60,
    "CZ": 50,
    "RZ": 60,
    "M": 60  # 测量操作
}
feature_names = ['num_qubits', 'num_gates',  'depth',  'program_communication', 'critical_depth', 'entanglement_ratio', 
             'parallelism', 'liveness', 'min_qubit_frequency', 'max_qubit_frequency', 'avg_qubit_frequency', 
             'min_readoutError', 'max_readoutError', 'avg_readoutError', 'min_singleQubit_gate_error', 'max_singleQubit_gate_error', 
             'avg_singleQubit_gate_error', 'min_twoQubits_gate_error', 'max_twoQubits_gate_error', 'avg_twoQubits_gate_error', 
             'min_t1', 'max_t1', 'avg_t1', 'min_t2', 'max_t2', 'avg_t2', 'twoQubits_gate_error_accumulation',  
             'idel_entropy', 'idle_rate','expected_fidelity']

def features_to_df(features):
    Xs = []
    for obj in features:
        X = []
        X.append(obj.num_qubits)
        X.append(obj.num_gates)
        X.append(obj.depth)
        X.append(obj.program_communication)
        X.append(obj.critical_depth)
        X.append(obj.entanglement_ratio)
        X.append(obj.parallelism)
        X.append(obj.liveness)
        X.append(obj.min_qubit_frequency)
        X.append(obj.max_qubit_frequency)
        X.append(obj.avg_qubit_frequency)
        X.append(obj.min_readoutError)
        X.append(obj.max_readoutError)
        X.append(obj.avg_readoutError)
        X.append(obj.min_singleQubit_gate_error)
        X.append(obj.max_singleQubit_gate_error)
        X.append(obj.avg_singleQubit_gate_error)
        X.append(obj.min_twoQubits_gate_error)
        X.append(obj.max_twoQubits_gate_error)
        X.append(obj.avg_twoQubits_gate_error)
        X.append(obj.min_t1)
        X.append(obj.max_t1)
        X.append(obj.avg_t1) 
        X.append(obj.min_t2)
        X.append(obj.max_t2)
        X.append(obj.avg_t2)         
        X.append(obj.twoQubits_gate_error_accumulation)                                
        X.append(obj.idel_entropy)
        X.append(obj.idle_rate)
        X.append(obj.expected_fidelity)
        Xs.append(X)
    # Xs = np.array(Xs)
    # convert 2D list to DataFrame
    xs = pd.DataFrame(Xs, columns=feature_names)
    return Xs

def get_Xs(props, qcis_str: str, layouts: list):
    
    if not validate_connectivity(qcis_str):
        print(f"there are too many optional layouts ##{len(layouts)}## for the current circuit, default chose 200")
    else:
        features = []
        for layout_id in range(len(layouts)):
            layout = layouts[layout_id]
            feature = calc_features(qcis_str, layout, props['backend'], props)
            features.append(feature)
        
    Xs = features_to_df(features=features)
    return Xs

class Features:
    def __init__(self, num_qubits, num_gates, depth, program_communication,
                 critical_depth, entanglement_ratio, parallelism, liveness, 
                 min_qubit_frequency, max_qubit_frequency, avg_qubit_frequency,
                 min_readout_frequency, max_readout_frequency, avg_readout_frequency,
                 min_readoutError, max_readoutError, avg_readoutError, 
                 min_singleQubit_gate_error, max_singleQubit_gate_error, avg_singleQubit_gate_error, 
                 min_twoQubits_gate_error, max_twoQubits_gate_error, avg_twoQubits_gate_error,
                 min_t1, max_t1, avg_t1, 
                 min_t2, max_t2, avg_t2,     
                 twoQubits_gate_error_accumulation, expected_fidelity, idel_entropy, idle_rate,
                 qubit_frequency_dict, readout_frequency_dict, readoutError_dict, singleQubit_gate_error_dict, 
                 twoQubits_gate_error_dict, t1_dict, t2_dict, 
                 singleQubit_update_time, readout_update_time, twoQubitGate_update_time,
                 origin_circuit, layout, transpiled_circit):
        
        self.num_qubits = num_qubits
        self.num_gates = num_gates
        self.depth = depth
        self.program_communication = program_communication
        self.critical_depth = critical_depth
        self.entanglement_ratio = entanglement_ratio
        self.parallelism = parallelism
        self.liveness = liveness
        self.min_qubit_frequency = min_qubit_frequency
        self.max_qubit_frequency = max_qubit_frequency 
        self.avg_qubit_frequency = avg_qubit_frequency
        self.min_readout_frequency = min_readout_frequency 
        self.max_readout_frequency = max_readout_frequency 
        self.avg_readout_frequency = avg_readout_frequency
        self.min_readoutError = min_readoutError 
        self.max_readoutError = max_readoutError 
        self.avg_readoutError = avg_readoutError 
        self.min_singleQubit_gate_error = min_singleQubit_gate_error 
        self.max_singleQubit_gate_error = max_singleQubit_gate_error 
        self.avg_singleQubit_gate_error = avg_singleQubit_gate_error
        self.min_twoQubits_gate_error = min_twoQubits_gate_error 
        self.max_twoQubits_gate_error = max_twoQubits_gate_error 
        self.avg_twoQubits_gate_error = avg_twoQubits_gate_error             
        self.min_t1 = min_t1
        self.max_t1 = max_t1
        self.avg_t1 = avg_t1
        self.min_t2 = min_t2
        self.max_t2 = max_t2
        self.avg_t2 = avg_t2
        self.twoQubits_gate_error_accumulation = twoQubits_gate_error_accumulation
        self.expected_fidelity = expected_fidelity
        self.idel_entropy = idel_entropy
        self.idle_rate = idle_rate      
        self.qubit_frequency_dict = qubit_frequency_dict if qubit_frequency_dict else None
        self.readout_frequency_dict = readout_frequency_dict if readout_frequency_dict else None
        self.readoutError_dict = readoutError_dict if readoutError_dict else None
        self.singleQubit_gate_error_dict = singleQubit_gate_error_dict if singleQubit_gate_error_dict else None
        self.twoQubits_gate_error_dict = twoQubits_gate_error_dict if twoQubits_gate_error_dict else None
        self.t1_dict = t1_dict if t1_dict else None
        self.t2_dict = t2_dict if t2_dict else None
        self.singleQubit_update_time = singleQubit_update_time if singleQubit_update_time else None
        self.readout_update_time = readout_update_time if readout_update_time else None
        self.twoQubitGate_update_time = twoQubitGate_update_time if twoQubitGate_update_time else None
        self.origin_circuit = origin_circuit
        self.layout = layout
        self.transpiled_circit = transpiled_circit

def filter_key_word(qcis_str, key_word):
    # 将 QCIS 代码按行分割
    lines = qcis_str.splitlines()
    # 过滤掉以 'B' 开头的行
    if isinstance(key_word, str):
        key_word = [key_word]
    if isinstance(key_word, list):
        key_word = tuple(key_word)
    filtered_lines = [
        line 
        for line in lines
        if not line.startswith(key_word)
    ]
    # 将处理后的行重新组合成新的 QCIS 代码字符串
    new_qcis_str = '\n'.join(filtered_lines)
    return new_qcis_str

def calc_dict(data):
    if any(isinstance(value, dict) for value in data.values()):
        value_list = [item['cz_error'] for item in list(data.values())]
        min_value = min(value_list)
        max_value = max(value_list)
        avg_value = sum(value_list) / len(value_list)
        return min_value, max_value, avg_value
    else:    
        if data:  # 确保字典不为空
            min_value = min(data.values())
            max_value = max(data.values())
            avg_value = sum(data.values()) / len(data)
        else:
            min_value, max_value, avg_value = None, None, None
    return min_value, max_value, avg_value

# calculate total execution time for each qubit
def calculate_qubit_times(quantum_circuit):
    qubit_times = {}
    for operation in quantum_circuit:
        gate = operation[0]
        time = gate_execution_times[gate] 
        if gate == "CZ":  # cz gate involves two qubits
            qubit1 = operation[1]
            qubit2 = operation[2]
            qubit_times[qubit1] = qubit_times.get(qubit1, 0) + time
            qubit_times[qubit2] = qubit_times.get(qubit2, 0) + time
        else:  # single qubit gate operation
            qubit = operation[1]
            qubit_times[qubit] = qubit_times.get(qubit, 0) + time
    return qubit_times

# calculate entropy
def calculate_entropy(idle_times, total_time):
    entropy = 0.0
    for idle_time in idle_times.values():
        p = idle_time / total_time
        if p > 0:
            entropy -= p * math.log2(p)  # entropy formula
    return entropy

# parse quantum circuit
def parse_quantum_circuit(description):
    quantum_circuit = []
    qubits = set()
    cz_list = []
    # split each line and parse them line by line
    for line in description.strip().splitlines():
        parts = line.split()

        # parse the gate and qubits involved in the gate
        gate = parts[0]  # gate type
        if gate == "CZ":  # if the gate is CZ, it involves two qubits
            qubit1 = parts[1]
            qubit2 = parts[2]
            quantum_circuit.append([gate, qubit1, qubit2])
            cz_list.append((int(re.search(r'\d+', qubit1).group()), int(re.search(r'\d+', qubit2).group())))
            qubits.update([qubit1, qubit2])
        elif gate =="B":
            quantum_circuit.append(parts)
        elif gate in {"Y2M", "X2M", "Y2P", "X2P", "RZ", "I", "XY2P", "XY2M", "M"}:  # 如果是单比特门或测量
            qubit = parts[1]
            quantum_circuit.append([gate, qubit])
            
            # if the gate is RZ, the third part is the phase parameter
            if gate == "RZ":
                parameter = parts[2]
                quantum_circuit[-1].append(float(parameter))
            qubits.add(qubit)
    return quantum_circuit, sorted(qubits), cz_list

# calculate idle times for each qubit
def calculate_idle_times(qubit_times, total_time, qubits):
    idle_times = {qubit: total_time - qubit_times.get(qubit, 0) for qubit in qubits}
    return idle_times

def calc_qubit_index(qubit, qubits):
    return qubits.index(qubit)

def get_involved_qubits(gate_info, qubits):
    # parse the gate and return the indices of qubits involved in the gate
    gate = gate_info[0]
    if gate_info[0] == "RZ":
        # if the gate is RZ, the qubit is the second element, and the third is the phase parameter
        involved_qubits = [calc_qubit_index(gate_info[1], qubits)]
    else:
        # otherwise, all subsequent elements are qubits
        involved_qubits = [calc_qubit_index(q, qubits) for q in gate_info[1:]]
    return involved_qubits


def calc_features(init_qcis: str, fixed_layout: list, backend: str, props: dict) -> Features:
    
    # parse the initial quantum circuit
    qc_depth = Circuit.load(filter_key_word(init_qcis, "B")).depth()
    
    # get the remapped quantum circuit
    replacement_rules = {}
    for i in range(len(fixed_layout)):      
        replacement_rules[f'Q{i}'] = f'Q{fixed_layout[i]}'
    submitted_qcis = replace_pattern(init_qcis, replacement_rules)    
    
    parsed_circuit, _, _ = parse_quantum_circuit(init_qcis)
    submitted_parsed_circuit, qubits, cz_list = parse_quantum_circuit(submitted_qcis)
    num_qubits = len(qubits)
    connectivity_collection = [[] for _ in range(num_qubits)]
    liveness_A_matrix = 0

    for gate_info in submitted_parsed_circuit:
        gate = gate_info[0]
        if gate in ("B", "M"):
            continue
        all_indices = get_involved_qubits(gate_info, qubits)
        liveness_A_matrix += len(all_indices)
        
        for qubit_index in all_indices:
            to_be_added_entries = all_indices.copy()
            to_be_added_entries.remove(int(qubit_index))
            connectivity_collection[int(qubit_index)].extend(to_be_added_entries)
        
    connectivity: list[int] = [len(set(connectivity_collection[i])) for i in range(num_qubits)]
    num_gates = sum(1 for gate_info in parsed_circuit if gate_info[0] not in ["B", "M"])
    num_multiple_qubit_gates = sum(1 for gate_info in parsed_circuit if gate_info[0] == "CZ")
    depth = Circuit.load(filter_key_word(init_qcis, ["B","M"])).depth()
    program_communication = np.sum(connectivity) / (num_qubits * (num_qubits - 1))
    if num_multiple_qubit_gates == 0:
        critical_depth = 0.0
    else:
        critical_depth = num_multiple_qubit_gates / num_multiple_qubit_gates    
    
    entanglement_ratio = num_multiple_qubit_gates / num_gates if num_gates else 0
    parallelism = (num_gates / depth - 1) / (num_qubits - 1) if num_qubits > 1 else 0
    liveness = liveness_A_matrix / (depth * num_qubits) if depth * num_qubits > 0 else 0
    
    twoQubits_gate_error_dict = {} 
    cz_error_list = []
    
    qubit_times = calculate_qubit_times(submitted_parsed_circuit)
    i = 0
    for item in cz_list:
        q0 = item[0]
        q1 = item[1]
        cz_error_list.append((q0,q1))
        twoQubits_gate_error_dict[i] = {}
        twoQubits_gate_error_dict[i]['twoQubits'] = (q0,q1)
        twoQubits_gate_error_dict[i]['cz_error'] = props['CZ_error'][(q0,q1)]
        i += 1
    
    qubit_frequency_dict = {}
    readout_frequency_dict = {}
    readoutError_dict = {}
    singleQubit_gate_error_dict = {}
    
    t1_dict = {} 
    t2_dict = {}
    for qubit_id in fixed_layout:
        qubit_frequency_dict[qubit_id] = props['qubit_frequency'][qubit_id]
        readout_frequency_dict[qubit_id] = props['readout_frequency'][qubit_id]
        readoutError_dict[qubit_id] = props['readoutError'][qubit_id]  # Baihua后端数据中没有读出误差数据，以单比特门误差近似替代
        singleQubit_gate_error_dict[qubit_id] = props['singleQubit_error'][qubit_id]
        t1_dict[qubit_id] = props['t1_dict'][qubit_id]
        t2_dict[qubit_id] = props['t2_dict'][qubit_id]    

    min_qubit_frequency, max_qubit_frequency, avg_qubit_frequency = calc_dict(qubit_frequency_dict)
    min_readout_frequency = max_readout_frequency = avg_readout_frequency = None
    min_readoutError, max_readoutError, avg_readoutError = calc_dict(readoutError_dict)
    min_singleQubit_gate_error, max_singleQubit_gate_error, avg_singleQubit_gate_error = calc_dict(singleQubit_gate_error_dict)
    min_twoQubits_gate_error, max_twoQubits_gate_error, avg_twoQubits_gate_error = calc_dict(twoQubits_gate_error_dict)
    min_t1, max_t1, avg_t1 = calc_dict(t1_dict)
    min_t2, max_t2, avg_t2 = calc_dict(t2_dict)            
    twoQubits_gate_error_accumulation = sum([item['cz_error'] for item in list(twoQubits_gate_error_dict.values())])
    expected_fidelity = 1 - default_cost(props, init_qcis, [fixed_layout], backend)[0][1]
    
    total_run_time = max(qubit_times.values())
    idle_times = calculate_idle_times(qubit_times, total_run_time, qubits)
    idel_entropy = calculate_entropy(idle_times, total_run_time)
    idle_rate = np.sum(list(idle_times.values())) / np.sum(list(qubit_times.values())) 
    
    singleQubit_update_time = props['singleQubit_update_time'] 
    readout_update_time = props['readout_update_time']
    twoQubitGate_update_time = props['twoQubitGate_update_time'] 
    
    return Features(num_qubits, num_gates, depth, program_communication,
        critical_depth, entanglement_ratio, parallelism, liveness, 
        min_qubit_frequency, max_qubit_frequency, avg_qubit_frequency,
        min_readout_frequency, max_readout_frequency, avg_readout_frequency,
        min_readoutError, max_readoutError, avg_readoutError, 
        min_singleQubit_gate_error, max_singleQubit_gate_error, avg_singleQubit_gate_error, 
        min_twoQubits_gate_error, max_twoQubits_gate_error, avg_twoQubits_gate_error,
        min_t1, max_t1, avg_t1, 
        min_t2, max_t2, avg_t2,
        twoQubits_gate_error_accumulation, expected_fidelity, idel_entropy, idle_rate,
        qubit_frequency_dict, readout_frequency_dict, readoutError_dict, singleQubit_gate_error_dict, 
        twoQubits_gate_error_dict, t1_dict, t2_dict, singleQubit_update_time, readout_update_time, twoQubitGate_update_time,
        init_qcis, fixed_layout, submitted_qcis)
    
if __name__ == '__main__':
    login_key = "xxxxx" #please apply for a login key from tianyan quantum computing platform
    machine_name = "tianyan176-2"
    props = tianyan_machine_config(login_key = login_key, backend = machine_name)
    layout = [44, 38, 45, 50, 56]
    init_qcis = r'''Y2M Q0
RZ Q0 3.14159265
RZ Q1 3.14159265
Y2P Q1
Y2M Q2
RZ Q2 3.14159265
RZ Q3 3.14159265
Y2P Q3
Y2M Q4
RZ Q4 3.14159265
CZ Q4 Q3
Y2M Q3
RZ Q3 3.14159265
CZ Q3 Q2
RZ Q2 3.14159265
Y2P Q2
CZ Q2 Q1
Y2M Q1
RZ Q1 3.14159265
CZ Q1 Q0
RZ Q0 3.14159265
Y2P Q0
B Q0 Q1 Q2 Q3 Q4
M Q0
M Q1
M Q2
M Q3
M Q4
'''
    features = calc_features(init_qcis, layout, machine_name, props)
    
    print(features.__dict__)