from datetime import datetime
import re, os, joblib, time, math, warnings, json, random
from qiskit.transpiler.coupling import CouplingMap
from qiskit.converters import circuit_to_dag
from cqlib.utils import QcisToQasm, QasmToQcis
from cqlib import TianYanPlatform  
from cqlib.utils import LaboratoryUtils
from qiskit import transpile
from qiskit import QuantumCircuit as qiskit_QuantumCircuit
from qiskit.quantum_info.analysis import hellinger_fidelity
from rustworkx import PyDiGraph, PyGraph, vf2_mapping
from tianyan_utils import  parse_quantum_circuit, evaluate_layouts
from tianyan_utils import FromStrToCircuitInstruction, replace_pattern, tianyan_machine_config, sort_key
from tianyan_utils import ml_predict, tedc_cost
from common import merge_dicts
from ml_features import get_Xs

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def nested_list_to_tuple_list(list_temp):
    new_list  = []
    for x in list_temp:
        new_list.append((x[0], x[1])) 
    return new_list

def readout_data_to_counts(result):
    #print(result)
    state01 = result["resultStatus"]
    # calculate the counts of each quantum state (key is a string)
    counts = {}
    for state in state01[1:]:  # skip the first item of list
        key = ''.join(map(str, state))
        if key not in counts.keys():
            counts[key] = 1
        else:
            counts[key] += 1  
    
    return counts

def qcis_circuit_result(login_key, machine_name: "tianyan_sw", circuit_qcis, shots: 1024, fixed_layout = None):
    lu = LaboratoryUtils()
    if machine_name == "tianyan_sw":
        qcis_str = circuit_qcis
        sim_platform = TianYanPlatform(login_key=login_key, machine_name= machine_name)
        lab_id = sim_platform.create_lab(name=f'lab.{datetime.now().strftime("%Y%m%d%H%M%S")}', remark='test_collection')
        query_id_single = sim_platform.submit_job(
                    circuit = qcis_str,
                    exp_name = f'exp.{datetime.now().strftime("%Y%m%d%H%M%S")}',
                    lab_id = lab_id,
                    num_shots=shots,
                )
        exp_result = sim_platform.query_experiment(query_id=query_id_single, max_wait_time=6000, sleep_time=5)
        result = lu.readout_data_to_state_probabilities_part(result=exp_result[0])
        return result
    
    elif machine_name in ["tianyan176", "tianyan24", "tianyan176-2"]:
        real_platform = TianYanPlatform(login_key=login_key, machine_name= machine_name)
        if fixed_layout:
            replacement_rules = {}
            for i in range(len(fixed_layout)):      
                replacement_rules[f'Q{i}'] = f'Q{fixed_layout[i]}'
            qcis_str = replace_pattern(circuit_qcis, replacement_rules)
            # print(qcis_str)
            lab_id = real_platform.create_lab(name=f'lab.{datetime.now().strftime("%Y%m%d%H%M%S")}', remark='test_collection')
            query_id_single = real_platform.submit_job(
                            circuit = qcis_str,
                            exp_name = f'exp.{datetime.now().strftime("%Y%m%d%H%M%S")}',
                            lab_id = lab_id,
                            num_shots=shots,
                        )
            # print(query_id_single)
            try:
                exp_result = real_platform.query_experiment(query_id=query_id_single, max_wait_time=6000, sleep_time=5)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None
            result = lu.readout_data_to_state_probabilities_part(result=exp_result[0])
            return result          
        else:
            qcis_str = circuit_qcis
            lab_id = real_platform.create_lab(name=f'lab.{datetime.now().strftime("%Y%m%d%H%M%S")}', remark='test_collection')
            query_id_single = real_platform.submit_job(
                        circuit = qcis_str,
                        exp_name = f'exp.{datetime.now().strftime("%Y%m%d%H%M%S")}',
                        lab_id = lab_id,
                        num_shots=shots,
                    )
            try:
                exp_result = real_platform.query_experiment(query_id=query_id_single, max_wait_time=6000, sleep_time=5)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None 
            result = lu.readout_data_to_state_probabilities_part(result=exp_result[0])
            return result
    
    elif machine_name == "qasm_simulator":
        trans_qc = qiskit_QuantumCircuit.from_qasm_str(circuit_qcis)     
        backend = Aer.get_backend('qasm_simulator')
        job = execute(trans_qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(trans_qc) 
        return counts
    
def convert_unmapping(qcis_str):
    qasm_str = QcisToQasm.convert_qcis_to_qasm(qcis_str)
    qcis_str = QasmToQcis().convert_to_qcis(qasm_str)
    pattern = r'^M Q(\d+)$'
    matches = re.findall(pattern, qcis_str, re.MULTILINE)
    # transform the matched strings to integers
    numbers = [int(match) for match in matches]
    replacement_rules = {}
    for i in range(len(numbers)):      
        replacement_rules[f'Q{numbers[i]}'] = f'Q{i}'
    qcis_str = replace_pattern(qcis_str, replacement_rules)
    return qcis_str, len(numbers)

def get_topology_from_qcis_str_and_props(props, qcis_str):
    #backend_topology = props['bidirectional_topography']
    backend_topology = props['topography']
    _, _, cz_list  = parse_quantum_circuit(qcis_str)
    return backend_topology, cz_list

def vf2_mapping_search_rust(props, qcis_str):
    
    qasm_str = QcisToQasm.convert_qcis_to_qasm(qcis_str)
    trans_qc = qiskit_QuantumCircuit.from_qasm_str(qasm_str)
    circ_qubits = trans_qc.num_qubits
    dag = circuit_to_dag(trans_qc)
    qubits = dag.qubits
    qubit_indices = {qubit: index for index, qubit in enumerate(qubits)}
    interactions = []
    for node in dag.op_nodes(include_directives=False):  # if include_directives=False, it menas include `barrier`, `snapshot` etc.
        len_args = len(node.qargs)
        if len_args == 2:
            interactions.append((qubit_indices[node.qargs[0]], qubit_indices[node.qargs[1]]))
    im_graph = PyGraph(multigraph=False)
    im_graph.add_nodes_from(range(len(qubits)))
    im_graph.add_edges_from_no_data(interactions)
    
    cm_graph_nodes = []
    for item in nested_list_to_tuple_list(props['topography']):
        if item[0] not in cm_graph_nodes:
            cm_graph_nodes.append(item[0])
        if item[1] not in cm_graph_nodes:
            cm_graph_nodes.append(item[1])

    cm_graph = PyGraph(multigraph=False)
    cm_graph.add_nodes_from(range(max(cm_graph_nodes)+1))
    unuesd_nodes = list(set([x for x in range(max(cm_graph_nodes)+1)]) - set(cm_graph_nodes))
    cm_graph.remove_nodes_from(unuesd_nodes)
    cm_graph.add_edges_from_no_data(nested_list_to_tuple_list(props['topography']))

    mappings = vf2_mapping(
        cm_graph,
        im_graph,
        subgraph=True,
        id_order=False,
        induced=False,
        call_limit=int(3e10)
    )

    layouts = []
    for mapping in mappings:
        temp_list = [None]*trans_qc.num_qubits
        for cm_i, im_i in mapping.items():
            key = qubits[im_i]
            val = cm_i
            temp_list[trans_qc.find_bit(key).index] = val
        layouts.append(temp_list)
    return layouts

def get_topology_from_qcis_str_and_props(props, qcis_str):
    backend_topology = props['bidirectional_topography']
    _, _, cz_list  = parse_quantum_circuit(qcis_str)
    return backend_topology, cz_list

def sabre_mapping_search(login_key, props, input_qcis_str, strategy, machine_name):
    qasm_str = QcisToQasm.convert_qcis_to_qasm(input_qcis_str)
    trans_qc = qiskit_QuantumCircuit.from_qasm_str(qasm_str)
    props = tianyan_machine_config(login_key, "tianyan176")
    CM = CouplingMap(props['bidirectional_topography'])
    circuit = transpile(trans_qc, coupling_map = CM, basis_gates=['h','cz', 'sxdg', 'ry', 'rz', 'sx'],
                        optimization_level=3)
    qcis_str = QasmToQcis().convert_to_qcis(circuit.qasm())
    selected_layout = get_best_layouts(props, qcis_str, strategy, machine_name)
    return selected_layout

def best_layout_with_pm(props, input_qcis_str, layouts, machine_name, topN =  None):
    layout_and_error = evaluate_layouts(props, input_qcis_str, layouts, machine_name)
    layout_and_error.sort(key=lambda x: x[1],reverse=False)
    #print(layout_and_error)
    if topN == None:
        return layout_and_error[0][0]
    else:
        topN = topN if topN < len(layout_and_error) else len(layout_and_error)
        selected_layouts = [layout_and_error[x][0] for x in range(topN)]
        return selected_layouts

def best_layout_with_tedqc(props, input_qcis_str, layouts, machine_name, topN =  None):
    layout_and_error = evaluate_layouts(props, input_qcis_str, layouts, machine_name, tedc_cost)
    layout_and_error.sort(key=lambda x: x[1],reverse=False)
    #print(layout_and_error)
    if topN == None:
        return layout_and_error[0][0]
    else:
        topN = topN if topN < len(layout_and_error) else len(layout_and_error)
        selected_layouts = [layout_and_error[x][0] for x in range(topN)]
        return selected_layouts
    
def best_layout_with_rs(props, input_qcis_str, layouts, machine_name, rs_num =  None):
    layout_and_error = evaluate_layouts(props, input_qcis_str, layouts, machine_name)
    layout_and_error.sort(key=lambda x: x[1],reverse=False)
    #print(layout_and_error)
    if rs_num == None:
        return random.sample(layout_and_error, 1)[0]
    else:
        rs_num = rs_num if rs_num < len(layout_and_error) else len(layout_and_error)
        selected_layouts = [x[0] for x in random.sample(layout_and_error, rs_num)]
        return selected_layouts

def best_layout_with_rf(props, input_qcis_str, layouts, machine_name, topN: None):
    #print(f"get all layouts cost {t2-t1} s")
    xs = get_Xs(props, input_qcis_str, layouts)

    t1 = time.time()
    y_pred_rf = ml_predict(xs, 'rf').tolist()
    t2 = time.time()
    #print(f"random forest inference cost {t2-t1} s")

    t1 = time.time()
    max_index = y_pred_rf.index(max(y_pred_rf))
    selected_layout = layouts[max_index]
    t2 = time.time()
    print(f"theoretical best layout is {selected_layout}, with estimated fidelity {max(y_pred_rf)}")
    
    return selected_layout

def best_layout_with_ml(props, input_qcis_str, layouts, machine_name, topN: None):

    t1 = time.time()
    xs = get_Xs(props, input_qcis_str, layouts)
    t2 = time.time()
    print(f"creating features cost {t2-t1} s")
    
    model_name = 'rf'
    t1 = time.time()
    y_pred_rf = ml_predict(xs, model_name).tolist()
    t2 = time.time()
    print(f"model inference cost {t2-t1} s")
    
    t1 = time.time()
    max_index = y_pred_rf.index(max(y_pred_rf))
    selected_layout = layouts[max_index]
    t2 = time.time()
    print(f"current model is {model_name}, the best layout is {selected_layout}, with estimated fidelity {max(y_pred_rf)}")

    return selected_layout

def get_best_layouts(props, input_qcis_str, strategy: "pm", machine_name, topN =  None):

    # t1 = time.time()
    layouts = vf2_mapping_search_rust(props=props, qcis_str=input_qcis_str)
    # t2 = time.time()
    if len(layouts) == 0:
        return None

    if strategy == "rf":
        selected_layouts = best_layout_with_rf(props, input_qcis_str, layouts, machine_name, topN)
    elif strategy == "pm":
        selected_layouts = best_layout_with_pm(props, input_qcis_str, layouts, machine_name, topN)
    elif strategy == "rs":
        selected_layouts = best_layout_with_rs(props, input_qcis_str, layouts, machine_name, topN)
    elif strategy == "tedqc":
        selected_layouts = best_layout_with_tedqc(props, input_qcis_str, layouts, machine_name, topN)
    elif  strategy == "ml":
        selected_layouts = best_layout_with_ml(props, input_qcis_str, layouts, machine_name, topN)      
    return selected_layouts   

def run_with_layouts(login_key, props, machine_name, tianyan_qc, layouts, shots = 8000, repeat = 1):
    if not isinstance(layouts[0], list):
        layouts = [layouts]
        
    all_layouts_num = len(vf2_mapping_search_rust(props, tianyan_qc))
    hellinger_fidelity_dict = {}
    hellinger_fidelity_dict["all_layouts_num"] = all_layouts_num
    for layout_index in range(len(layouts)):
        ERROR_EXIST = False
        layout = layouts[layout_index]
        hellinger_fidelity_dict[layout_index] = {}
        print(f"now the layout is {layout}\n")

        layout_and_error = evaluate_layouts(props, tianyan_qc, layout, machine_name)  
        error = layout_and_error[0][1]
        hf_list = []
        exec_time = 0
        for i in range(repeat):
            start_time = time.time()
            try:
                real_result = qcis_circuit_result(login_key, machine_name, tianyan_qc, shots, fixed_layout = layout) #filter_key_word(tianyan_qc,"B")
            except Exception as e:
                print(f"An unexpected error occurred: {e}") 
                hellinger_fidelity_dict[layout_index]['layout'] = layout
                hellinger_fidelity_dict[layout_index]['expected_fidelity'] = None
                hellinger_fidelity_dict[layout_index]['exec_fidelity'] = None
                ERROR_EXIST = True
                break
                
            # print(f"real_result get")
            sim_result = qcis_circuit_result(login_key, 'tianyan_sw', tianyan_qc, shots)
            # print(f"sim_result get\n")
            end_time = time.time()
            print(f"######repeat round {i} costs {end_time-start_time:.6f}s ######")
            # print("Now the time is",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            hf = hellinger_fidelity(sim_result, real_result)
            hf_list.append(hf)
            exec_time += end_time-start_time
        
        if ERROR_EXIST:
            continue

        hf = sum(hf_list) / len(hf_list)
        print(layout, hf, 1 - error, "\n")  
        hellinger_fidelity_dict[layout_index]['layout'] = layout
        hellinger_fidelity_dict[layout_index]['expected_fidelity'] = 1 - error
        hellinger_fidelity_dict[layout_index]['exec_fidelity'] = hf
    return hellinger_fidelity_dict

def run_with_single_layout(login_key, props, machine_name, tianyan_qc, layout, shots = 8192, repeat =  2):
    hellinger_fidelity_dict = {}
    error = evaluate_layouts(props, tianyan_qc, layout, machine_name)[0][1]
    sim_result_list = []
    real_result_list = []
    exec_time = 0
    for i in range(repeat):
        # hellinger_fidelity_list[i] = {}
        start_time = time.time()
        # print(f"\n######repeat round {i}######")
        try:
            real_result = qcis_circuit_result(login_key, machine_name, tianyan_qc, shots, fixed_layout = layout) 
            if real_result == None:
                return layout, 1 - error, None
            real_result_list.append(real_result)
        except Exception as e:
            print(f"An unexpected error occurred: {e}") 
            return layout, 1 - error, None
            
        # print(f"real_result get")
        # print(f"{tianyan_qc}")
        sim_result = qcis_circuit_result(login_key, 'tianyan_sw', tianyan_qc, shots)
        sim_result_list.append(sim_result)    
        # print(f"sim_result get\n")
        end_time = time.time()
        print(f"######repeat round {i} costs {end_time-start_time:.6f}s ######")
        # print("Now the time is",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    post_sim_result = merge_dicts(sim_result_list)
    post_real_result = merge_dicts(real_result_list)
    hf = hellinger_fidelity(post_sim_result, post_real_result)
    print(layout, 1 - error, hf, "\n")

    return layout,  1 - error, hf

if __name__ == '__main__':
    
    
    login_key = "xxxxx" #please apply for a login key from tianyan quantum computing platform
    machine_name = "tianyan176-2"
    repeat = 3
    shots = 8000
    strategy = "pm"
    props = tianyan_machine_config(login_key = login_key, backend = machine_name)
    print(props)

    qcis_dir = "./Benchmarks/tianyan_benchmark/random_circuits/"

    file_path_list = []
    for filename in os.listdir(qcis_dir):
        if filename.endswith(".qcis"):
            file_path = os.path.join(qcis_dir, filename)
            file_path_list.append(file_path)
    file_path_list = sorted(file_path_list, key=sort_key)
    for file_path in file_path_list[6:7]:
        with open(file_path, 'r') as f:
            input_qcis_str = f.read()
        print(file_path)
        
        best_layout = get_best_layouts(props, input_qcis_str, strategy, machine_name)
        print(f"get layout method: {strategy}")
        print(best_layout)
        
        hellinger_fidelity_list = run_with_layout(login_key, machine_name, input_qcis_str, best_layout, shots, repeat)
        print(json.dumps(hellinger_fidelity_list, indent=4))

