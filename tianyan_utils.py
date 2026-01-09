import re, os, joblib, time, math, warnings
from collections import defaultdict, deque
import numpy as np
from datetime import datetime
import pandas as pd
from cqlib import TianYanPlatform  # import the SDK of TianYan Platform
from cqlib.utils import LaboratoryUtils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

def ml_predict(xs, model_name):
    model_dirs = {  'catBoost':'./model/CatBoost.joblib',
                    'rf':'./model/RF.joblib',
                    'svr':'./model/SVR.joblib',
                    'knn':'./model/KNN.joblib',
                    'dt':'./model/DT.joblib',
                    'gbdt':'./model/gbdt_model.joblib',
                    'lr':'./model/LR.joblib',
                    'xgboost':'./model/XGBoost.joblib',
                    'lightgbm':'./model/LightGBM.joblib'
                    }
    # load the model
    ml_model = joblib.load(model_dirs[model_name])
    y_pred_rf = ml_model.predict(xs)
    return y_pred_rf

def tianyan_topography(machine_name):
    topography_rename = {}
    if machine_name == "tianyan176-2" or machine_name == "tianyan176":
        k = 0
        for i in range(5):
            for j in range(6):
                if j<=4:
                    topography_rename[(12*i+j,12*i+j+6)] = "G"+str(k)
                    topography_rename[(12*i+j,12*i+j+7)] = "G"+str(k+1)
                    k+=2
                else:
                    topography_rename[(12*i+j,12*i+j+6)] = "G"+str(k)
                    k+=1
            for j in range(6):
                if j<=4:
                    topography_rename[(12*i+j+6,12*i+j+12)] = "G"+str(k)
                    topography_rename[(12*i+j+7,12*i+j+12)] = "G"+str(k+1)
                    k+=2
                else:
                    topography_rename[(12*i+j+6,12*i+j+12)] = "G"+str(k)
                    k+=1
        return topography_rename
    elif machine_name == "tianyan24":
        k = 0
        for i in range(1):
            for j in range(11):
                topography_rename[(12*i+j,12*i+j+1)] = "G"+str(k)
                topography_rename[(12*i+j,12*i+j+12)] = "G"+str(k+11)
                k+=1
            topography_rename[(12*i+11,12*i+11+12)] = "G"+str(k+11)
            k+=1
        k = k+11
        for i in range(1,2):
            for j in range(11):
                topography_rename[(12*i+j,12*i+j+1)] = "G"+str(k)
                k+=1       
        return topography_rename    

def tianyan_machine_config(login_key, backend: "tianyan176"):
    platform = TianYanPlatform(login_key=login_key, machine_name = backend)
    lu = LaboratoryUtils()
    config_save = platform.download_config()
    coupling_map =  lu.get_coupling_map(config_save)
    topography = [[item[1], item[0]] if item[0] > item[1] else [item[0], item[1]] for item in coupling_map]
    topography_rename = tianyan_topography(backend)
    props = {}
    CZ_error = config_save['twoQubitGate']['czGate']['gate error']['param_list'] #CZ错误率
    CZ_name = config_save['twoQubitGate']['czGate']['gate error']['qubit_used']  #相关的CZ门名称，与topography_rename相对应
    CZ_error_list = {}  # cz gate_error dict
    for key, value in topography_rename.items():
        if value in CZ_name:
            CZ_error_list[(key[0],key[1])] = CZ_error[CZ_name.index(value)]
            CZ_error_list[(key[1],key[0])] = CZ_error[CZ_name.index(value)]
        else:
            CZ_error_list[(key[0],key[1])] = np.nan
            CZ_error_list[(key[1],key[0])] = np.nan

    singleQubit_error_dict = {}
    singleQubit_readoutError_dict = {}
    qubit_frequency_dict = {}
    readout_frequency_dict = {}
    singleQubit_error = config_save['qubit']['singleQubit']['gate error']['param_list']
    singleQubit_list = config_save['qubit']['singleQubit']['gate error']['qubit_used']
    singleQubit_readoutErrorlist = config_save['readout']['readoutArray']['Readout Error']['param_list']
    qubit_frequency_list = config_save['qubit']['frequency']['f01']['param_list']
    
    # parse the qubit number in the quantum circuit, and convert it to integer type
    import re
    # regular expression, match all forms of numbers
    pattern1 = r'\d+'
    for i in range(len(singleQubit_list)):
        number = int(re.findall(pattern1, singleQubit_list[i])[0])
        singleQubit_error_dict[number] = singleQubit_error[i]
        singleQubit_readoutError_dict[number] = singleQubit_readoutErrorlist[i]
        qubit_frequency_dict[number] = qubit_frequency_list[i]
        readout_frequency_dict[number] = None
    # singleQubit_error_dict 
    props['backend'] = backend
    props['singleQubit_error'] = singleQubit_error_dict 
    # singleQubit_readoutError_dict
    props['readoutError'] = singleQubit_readoutError_dict 
    props['CZ_error'] = CZ_error_list
    props['qubit_frequency'] = qubit_frequency_dict
    props['readout_frequency'] = readout_frequency_dict
    sub_list = []  # 无效边列表
    for edge in topography:
        if props['CZ_error'][(edge[0],edge[1])] is np.nan:
            sub_list.append([edge[0],edge[1]])

    topography = [x for x in topography if x not in sub_list]

    node_list = []
    for item in topography:
        node_list.append(item[0])
        node_list.append(item[1])
    from collections import Counter
    node_dict = dict(Counter(node_list))

    dissociate_list = [] # isolated edges
    for edge in topography:
        if node_dict[edge[0]] < 2:
            if node_dict[edge[0]] == node_dict[edge[1]]:
                dissociate_list.append(edge)

    topography = [x for x in topography if x not in dissociate_list]

    graph_nodes = []
    for item in topography:
        graph_nodes.append(item[0])
        graph_nodes.append(item[1])

    graph_nodes = list(set(graph_nodes))
    
    bidirectional_topography = []
    for edge in topography:
        bidirectional_topography.append(edge)
        bidirectional_topography.append([edge[1],edge[0]])
    bidirectional_topography = sorted(bidirectional_topography)  

    props['topography'] = topography
    props['bidirectional_topography'] = bidirectional_topography
    props['singleQubit_update_time'] = config_save['qubit']['singleQubit']['gate error']['update_time'] #单比特门校准时间
    props['readout_update_time'] = config_save['readout']['readoutArray']['Readout Error']['update_time'] #读取错误率校准时间
    props['twoQubitGate_update_time'] = config_save['twoQubitGate']['czGate']['gate error']['update_time'] #双比特门校准时间  

    # t1、t2 time unit: us
    props['t1_dict'] = {}
    index = 0
    for item in config_save['qubit']['relatime']['T1']['qubit_used']:
        number = int(re.findall(pattern1, item)[0])
        props['t1_dict'][number] = config_save['qubit']['relatime']['T1']['param_list'][index]
        index+=1

    props['t2_dict'] = {}
    index = 0
    for item in config_save['qubit']['relatime']['T2']['qubit_used']:
        number = int(re.findall(pattern1, item)[0])
        props['t2_dict'][number] = config_save['qubit']['relatime']['T2']['param_list'][index]
        index+=1
    
    return props 

# define the execution time of each quantum gate, unit: ns
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
    "M": 60  # measurement operation
}

def replace_pattern(input_str, rules):
    pattern = re.compile(r'\b(Q[0-9]+)\b')
    def replace_qubits(match):
        qubit = match.group(1)
        return rules.get(qubit, qubit)
    output_string = pattern.sub(replace_qubits, input_str)
    return output_string

# custom sort key function
def sort_key(item):
    # extract the number part from the name field
    number = int(''.join(filter(str.isdigit, os.path.basename(item))))
    return number

def get_new_qcis_str_by_convert_with_layout(qcis_str, layout):
    replacement_rules = {}
    for i in range(len(layout)):      
        replacement_rules[f'Q{i}'] = f'Q{layout[i]}'
    new_qcis_str = replace_pattern(qcis_str, replacement_rules)
    return new_qcis_str

# calculate the total time each qubit is occupied
def calculate_qubit_times(quantum_circuit):
    qubit_times = {}
    for operation in quantum_circuit:
        gate = operation[0]
        time = gate_execution_times[gate]
        
        if gate == "CZ":  # CZ 门涉及两个量子比特
            qubit1 = operation[1]
            qubit2 = operation[2]
            qubit_times[qubit1] = qubit_times.get(qubit1, 0) + time
            qubit_times[qubit2] = qubit_times.get(qubit2, 0) + time
        else:  # 单比特门操作
            qubit = operation[1]
            qubit_times[qubit] = qubit_times.get(qubit, 0) + time
    return qubit_times

# calculate the entropy
def calculate_entropy(idle_times, total_time):
    entropy = 0.0
    for idle_time in idle_times.values():
        p = idle_time / total_time
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

# parse the quantum circuit
def parse_quantum_circuit(description):
    quantum_circuit = []
    qubits = set()
    cz_list = []
    # 分割每一行，并逐行解析
    # print(description)
    for line in description.strip().splitlines():
        parts = line.split()

        # 根据门的类型进行解析
        gate = parts[0]  # 门的类型
        if gate == "CZ":  # 如果是 CZ 操作，涉及两个量子比特
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
            
            # 如果是 RZ 操作，第三个部分是参数
            if gate == "RZ":
                parameter = parts[2]
                quantum_circuit[-1].append(float(parameter))

            qubits.add(qubit)
    return quantum_circuit, sorted(qubits), cz_list

def sort_and_vectorize_percentages(gate_percentages):
    # sort the gate types alphabetically
    sorted_gates = sorted(gate_percentages.keys())
    # 向量化成列表
    percentage_vector = [gate_percentages[gate] for gate in sorted_gates]
    return sorted_gates, percentage_vector

# calculate the idle time of each qubit
def calculate_idle_times(qubit_times, total_time, qubits):
    idle_times = {qubit: total_time - qubit_times.get(qubit, 0) for qubit in qubits}
    return idle_times

def calc_qubit_index(qubit, qubits):
    return qubits.index(qubit)

def get_involved_qubits(gate_info, qubits):
    # extract the qubits involved in the gate operation
    gate = gate_info[0]
    if gate_info[0] == "RZ":
        # if it is a RZ gate, the qubit is the second element, and the phase parameter is the third
        involved_qubits = [calc_qubit_index(gate_info[1], qubits)]
    else:
        # otherwise, all subsequent elements are qubits
        involved_qubits = [calc_qubit_index(q, qubits) for q in gate_info[1:]]
    return involved_qubits

# build the dependency graph
def build_dependency_graph(quantum_circuit):
    graph = defaultdict(list)
    last_operation_per_qubit = {}

    for index, operation in enumerate(quantum_circuit):
        gate, *qubits = operation
        if gate == "CZ":
            qubit1, qubit2 = qubits
            for qubit in [qubit1, qubit2]:
                if qubit in last_operation_per_qubit:
                    graph[last_operation_per_qubit[qubit]].append(index)
                last_operation_per_qubit[qubit] = index
        else:
            qubit = qubits[0]
            if qubit in last_operation_per_qubit:
                graph[last_operation_per_qubit[qubit]].append(index)
            last_operation_per_qubit[qubit] = index

    return graph

def calculate_time_difference(time_str1, time_str2, time_format="%Y-%m-%d %H:%M:%S"):

    # convert the strings to datetime objects
    time_obj1 = datetime.strptime(time_str1, time_format)
    time_obj2 = datetime.strptime(time_str2, time_format)
    # calculate the time difference
    time_diff = time_obj2 - time_obj1
    # convert the time difference to hours
    hours_diff = time_diff.total_seconds() / 3600
    return hours_diff

def evaluate_layouts(props, circ, layouts, backend, cost_function=None, TEST = False):
    if not any(layouts):
        return []
    if not isinstance(layouts[0], list):
        layouts = [layouts]
    if cost_function is None:
        cost_function = default_cost
#     out = cost_function(circ, layouts, backend)
    out = cost_function(props, circ, layouts, backend, TEST)
    out.sort(key=lambda x: x[1])
    return out

def default_cost(props, qcis_str, layouts, backend = None, TEST = False):
    if backend in ['tianyan176', 'tianyan176-2', 'tianyan24', 'tianyan504']:
        out = []
        check_double_list(layouts)
        for layout in layouts:
            if TEST:    ###set TEST to True to print the log
                print(layout)
            error = 0
            fid = 1
            replacement_rules = {}
            for i in range(len(layout)):      
                replacement_rules[f'Q{i}'] = f'Q{layout[i]}'
            qcis_layout_str = replace_pattern(qcis_str, replacement_rules)
            qcis_instructions = FromStrToCircuitInstruction(qcis_layout_str)
            for item in qcis_instructions:
                if TEST:
                    print("Now the operation is", item)
                if item.operation_name in ['cz']:
                    q0 = item.qubits[0]
                    q1 = item.qubits[1]
                    fid *= (1-props['CZ_error'][(q0,q1)]*0.01)
                    if TEST:
                        print(item.operation_name,props['CZ_error'][(q0,q1)]*0.01, fid)

                elif item.operation_name in ['x2p','x2m','y2p','y2m','rz']:
                    q0 = item.qubits[0]
                    fid *= (1-props['singleQubit_error'][q0]*0.01)
                    if TEST:
                        print(item.operation_name,props['singleQubit_error'][q0]*0.01, fid)

                elif item.operation_name in ['measure', 'reset']:
                    q0 = item.qubits[0]
                    fid *= (1-props['readoutError'][q0]*0.01)
                    if TEST:
                        print(item.operation_name,props['readoutError'][q0]*0.01, fid)
            if TEST:
                print("next layout")
            error = 1-fid
            out.append((layout, error))
        return out

def tedc_cost(props, qcis_str, layouts, backend = None, TEST = False):
    '''Computes the total error probability of a quantum circuit and the time needed to execute it.'''
    if backend in ['tianyan176', 'tianyan176-2', 'tianyan24', 'tianyan504']:
        out = []
        for layout in layouts:
            if TEST:    ###set TEST to True to print the log
                print(layout) 
            fidel_cz, fidel_sgate, fidel_measurement, fidel_time = 1, 1, 1, 1
            normal_gatetimes = 60   # ns
            cz_gatetimes = 50       # ns
            qubit_list = layout
            time = [0 for _ in qubit_list]
            fidel_time = [1 for _ in qubit_list] 

            # props t1 t2 unit: us, while gatetime unit: ns, need to convert t1, t2 to ns
            tdecay = [[props['t1_dict'][qubit]*1000, props['t2_dict'][qubit]*1000] for qubit in qubit_list ]

            replacement_rules = {}
            for i in range(len(layout)):      
                replacement_rules[f'Q{i}'] = f'Q{layout[i]}'
            qcis_layout_str = replace_pattern(qcis_str, replacement_rules)
            qcis_instructions = FromStrToCircuitInstruction(qcis_layout_str)
            #print('Computing gate error')
            for item in qcis_instructions:
                gate = item.operation_name
                q0 = item.qubits[0]
                idx_q0 = qubit_list.index(q0)
                if gate == 'cz':
                    q1 = item.qubits[1]
                    idx_q1 = qubit_list.index(q1)
                    gate_time = cz_gatetimes
                    time[idx_q0] += gate_time
                    time[idx_q1] += gate_time
                    time[idx_q0] = max([time[idx_q0], time[idx_q1]])
                    time[idx_q1] = max([time[idx_q0], time[idx_q1]])
                    cz_error = props['CZ_error'][(q0,q1)]*0.01
                    # compute new commulative error from entangled/control qubits
                    # # We understand : q0 = control , q1 = target
                    fidel_time[idx_q0] = np.exp(-time[idx_q0]/tdecay[idx_q0][0])*np.exp(-time[idx_q0]/tdecay[idx_q0][1])
                    fidel_time[idx_q1] = np.exp(-time[idx_q1]/tdecay[idx_q1][0])*np.exp(-time[idx_q1]/tdecay[idx_q1][1])

                elif gate == 'measure':
                    gate_time = normal_gatetimes
                    measurement_error = props['readoutError'][q0]*0.01
                    fidel_measurement *= (1 - measurement_error)
                    fidel_time[idx_q0] = np.exp(-time[idx_q0]/tdecay[idx_q0][0])*np.exp(-time[idx_q0]/tdecay[idx_q0][1])

                else:
                    if gate != 'barrier':             
                        gate_time = normal_gatetimes
                        time[idx_q0] += gate_time
                        sgate_error = props['singleQubit_error'][q0]*0.01
                        fidel_sgate *= (1 - sgate_error)
                        fidel_time[idx_q0] = np.exp(-time[idx_q0]/tdecay[idx_q0][0])*np.exp(-time[idx_q0]/tdecay[idx_q0][1])

            fidel_total = fidel_cz * fidel_sgate * np.prod(fidel_time) * fidel_measurement
            
            if TEST:
                print("next layout")
            error =  1 - fidel_total
            out.append((layout, error))
        return out

def check_double_list(layouts):
    # judge layouts is a list
    assert isinstance(layouts, list), "layouts must be a list, got {} instead".format(type(layouts).__name__)
    # judge each element in layouts is a list
    for element in layouts:
        # assert each element is a list
        assert isinstance(element, list), "all elements within layouts must be lists, found {}".format(type(element).__name__)

class CircuitInstruction():
    def __init__(
        self,
        operation_name: None,
        qubits: None,
        clbits: None,
    ):
        self.operation_name = operation_name
        self.qubits = qubits
        self.clbits = clbits
    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"operation_name={self.operation_name!r}"
            f", qubits={self.qubits!r}"
            f", clbits={self.clbits!r}"
            ")"
        )

def FromStrToCircuitInstruction(qcis_str):
    pattern = r'\d+'
    instruction = []
    # split the string into lines
    lines = qcis_str.split('\n')
    # traverse each line and print it
    for line in lines:
        operation = line.split(' ')
        if operation[0] in ['RZ']:
            qubits = [int(re.findall(pattern, operation[1])[0])]
            clbits = None
            operation = CircuitInstruction('rz', qubits, clbits)
            instruction.append(operation)
        elif operation[0] in ['CZ']:
            qubits = (int(re.findall(pattern, operation[1])[0]), int(re.findall(pattern, operation[2])[0]))
            clbits = None
            operation = CircuitInstruction('cz', qubits, clbits)
            instruction.append(operation)
        elif operation[0] in ['M']:
            qubits = [int(re.findall(pattern, operation[1])[0])]
            clbits = qubits
            operation = CircuitInstruction('measure', qubits, clbits)
            instruction.append(operation)
        elif operation[0] in ['X2P','X2M','Y2P','Y2M']:
            qubits = [int(re.findall(pattern, operation[1])[0])]
            clbits = None
            operation = CircuitInstruction(operation[0].lower(), qubits, clbits)
            instruction.append(operation)
    return instruction  

