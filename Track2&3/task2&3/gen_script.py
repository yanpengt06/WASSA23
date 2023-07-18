import yaml
import itertools
import sys

def yaml_2_slurm(yaml_path):
    
    with open(yaml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    program = data['program']
    method = data['method']
    parameters = data['parameters']
    para_value_list = []
    search_parameters = {}
    for para_name in parameters.keys():
        para_value = parameters[para_name]
        if 'value' in para_value.keys():
            para_value_list.append(f"--{para_name}={para_value['value']}")
            pass
        elif 'values' in para_value.keys():
            search_parameters[para_name] = para_value['values']
        else:
            raise NotImplementedError
    if method == 'grid':
        # 获取所有参数的排列组合
        param_combinations = list(itertools.product(*search_parameters.values()))

        # 将参数组合转换为字典形式
        search_param_list = []
        for combination in param_combinations:
            param = []
            for i, value in enumerate(combination):
                param_name = list(search_parameters.keys())[i]
                param.append(f"--{param_name}={value}")
            search_param_list.append(param)
    else:
        raise NotImplementedError
    
    for search_param in search_param_list:
        all_param = para_value_list + search_param
        print(f"python run.py {' '.join(all_param)}\n")
        

yaml_file_path = sys.argv[1]
print(yaml_file_path)

yaml_2_slurm(yaml_file_path)