import os,json,random
from tianyan_utils import tianyan_machine_config
from tianyan_platform import run_with_single_layout, get_best_layouts, run_with_layouts, vf2_mapping_search_rust
from common import PCOLOR
from ml_features import calc_features
from datetime import datetime
from common import get_execute_config, sort_key
import argparse

def machine_config(login_key, backend):
    if backend in ["tianyan176", "tianyan176-2", "tianyan24", "tianyan504"]:
        props = tianyan_machine_config(login_key, backend)
    return props

def remapping_with_pm(props, backend, input_circuit_str, topN = 1):
    if backend in ["tianyan176", "tianyan176-2", "tianyan24", "tianyan504"]:
        layouts = get_best_layouts(props, input_circuit_str, "pm", backend, topN)
    return layouts

def remapping_with_tedqc(props, backend, input_circuit_str, topN = 1):
    if backend in ["tianyan176", "tianyan176-2", "tianyan24", "tianyan504"]:
        layouts = get_best_layouts(props, input_circuit_str, "tedqc", backend, topN)
    return layouts

def remapping_with_rs(props, backend, input_circuit_str, rs_num = 1): 
    if backend in ["tianyan176", "tianyan176-2", "tianyan24", "tianyan504"]:
        layouts = get_best_layouts(props, input_circuit_str, "rs", backend, topN)

    return layouts

def remapping_with_ml(props, backend, input_circuit_str, topN = 1): 
    if backend in ["tianyan176", "tianyan176-2", "tianyan24", "tianyan504"]:
        layouts = get_best_layouts(props, input_circuit_str, "ml", backend, topN)
    return layouts

def remapping(backend, props, input_circuit_str, strategy, topN):
    # print(props)
    if strategy == "pm":
        new_layout = remapping_with_pm(props, backend, input_circuit_str, topN)
        return new_layout
    elif strategy == "rs":
        new_layout = remapping_with_rs(props, backend, input_circuit_str, topN)
        return new_layout
    elif strategy == "tedqc":
        new_layout = remapping_with_tedqc(props, backend, input_circuit_str, topN)
        return new_layout
    elif strategy == "ml":
        new_layout = remapping_with_ml(props, backend, input_circuit_str, topN)
        return new_layout

def run_single_strategy(login_key, backend, props, input_circuit_str, strategy, shots, repeat, topN):
    pcolor = PCOLOR()
    if strategy in ['pm','rs','tedqc','ml']: 
        expected_layouts = remapping(backend, props, input_circuit_str, strategy, topN)
        print(f'''get layout method: {pcolor.RED_BOLD}{strategy}{pcolor.RESET}
shots: {pcolor.RED_BOLD}{shots}{pcolor.RESET}
the expected_layout:{pcolor.RED_BOLD}{expected_layouts}{pcolor.RESET}''')
        hellinger_fidelity_dict = run_with_layouts(login_key, props, backend, input_circuit_str, expected_layouts, shots, repeat)
        print(f"expected_fidelity_dict:\n{json.dumps(hellinger_fidelity_dict, indent=4)}\n\n")  
        return hellinger_fidelity_dict
    elif strategy in ['simulator']:
        pass

def run_all_strategy(login_key, backend, props, input_circuit_str, shots, repeat, topN):
    exec_info = {}
    for strategy in ['ml','pm','rs','tedqc']:          #['pm','rs','tedqc']
        hellinger_fidelity_dict = run_single_strategy(login_key, backend, props, input_circuit_str, strategy, shots, repeat, topN)
        exec_info[strategy] = hellinger_fidelity_dict
    return exec_info

def generate_train_data(login_key, backend, props, repeat):
    pcolor = PCOLOR()
    
    execute_config = {}
    total_num = 0  
    ml_benchmarks_path = f"./qcis_and_execution_data/random_circuits/"
    file_path_list = []
    for filename in os.listdir(ml_benchmarks_path):
        if filename.endswith(".qcis"):
            file_path = os.path.join(ml_benchmarks_path, filename)
            file_path_list.append(file_path)
    file_path_list = sorted(file_path_list, key=sort_key)
    for file_path in file_path_list[:]:
        circuit_fidelity_result = {}
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        circuit_fidelity_result[file_name] = {}
        with open(file_path, 'r') as file:
            small_qc = file.read()
        layouts = vf2_mapping_search_rust(props, small_qc)
        
        if layouts == None:
            circuit_fidelity_result[file_name]['run_config'] = None
            circuit_fidelity_result[file_name]['circuit_features'] = None
            continue
        
        print(file_path, file_path_list.index(file_path))
        if len(layouts)>1000:
            print(f"there are too many optional layouts ##{len(layouts)}## for the current circuit, default chose 200")
            layouts = random.sample(layouts, 200)
            total_num += 200
        else:
            print(f"there are totally {len(layouts)} layouts")
            total_num += len(layouts)
        
        for layout_id in range(len(layouts)):
            props = machine_config(login_key = login_key, backend = backend)
            layout = layouts[layout_id]
            print(file_path, file_path_list.index(file_path))
            print(f"total layouts num is {len(layouts)}, now {pcolor.RED_BOLD}the layout is {layout}, index: {layout_id}{pcolor.RESET}\n")
            begin_time = datetime.now()
            layout, expected_fidelity, real_fidelity = run_with_single_layout(login_key, props, backend, small_qc, layout, shots, repeat)
            end_time = datetime.now()
            run_config = get_execute_config(backend, repeat, shots, layout, layout_id, expected_fidelity, real_fidelity, begin_time, end_time)
            circuit_fidelity_result[file_name]['run_config'] = run_config
            circuit_fidelity_result[file_name]['circuit_features'] = calc_features(small_qc, layout, backend, props).__dict__
            print(json.dumps(circuit_fidelity_result, indent=4))

            # save to file
            if real_fidelity == None:
                with open(f'./qcis_and_execution_data/random_circuits_data/{file_name}_data_{layout_id}_None.json', 'w') as f:
                    json.dump(circuit_fidelity_result, f, indent=4)
            else:
                with open(f'./qcis_and_execution_data/random_circuits_data/{file_name}_data_{layout_id}.json', 'w') as f:
                    json.dump(circuit_fidelity_result, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Quantum Layout Optimization Framework")
    parser.add_argument(
        '-generate',
        action='store_true',
        help='Generate training data'
    )
    parser.add_argument(
        '-test',
        action='store_true',
        help='Test different layout selection methods'
    )
    args = parser.parse_args()

    # 检查是否至少提供了一个选项
    if not (args.generate or args.test):
        print("Error: Please specify either -generate or -test.")
        print("Usage: python main.py -generate   # to generate training data")
        print("       python main.py -test      # to test layout selection methods")
        exit(0)


    login_key = "xxxxx"  # please apply the key from Tianyan Quantum Computing Cloud Platform
    
    backend = "tianyan176-2"
    repeat = 1
    strategy = "pm"
    shots = 8000
    topN = 1
    props = machine_config(login_key = login_key, backend = backend)
    print(props)

    if args.generate:
        # # 1.generate training data
        generate_train_data(login_key, backend, props, repeat)
    elif args.test:
        # # 2.test different layout selection methods
        intput_files_dir = "./qcis_and_execution_data/selected_circuits/"

        file_path_list = []
        for filename in os.listdir(intput_files_dir):
            if filename.endswith(".qcis"):
                file_path = os.path.join(intput_files_dir, filename)
                file_path_list.append(file_path)
            elif filename.endswith(".qasm"):
                file_path = os.path.join(intput_files_dir, filename)
                file_path_list.append(file_path)

        file_path_list = sorted(file_path_list, key=sort_key)
        qcis_exec_info = {}
        exec_time = f'run_used_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        for file_path in file_path_list[:]:
            with open(file_path, 'r') as f:
                input_circuit_str = f.read()
            print(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            exec_info = run_all_strategy(login_key, backend, props, input_circuit_str, shots, repeat, topN)
            qcis_exec_info[file_name] = exec_info
        with open(f'./{exec_time}.json', 'w') as f:
            json.dump(qcis_exec_info, f, indent=4)
        
