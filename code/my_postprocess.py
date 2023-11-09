import re
import my_tool as mtool
import pandas as pd

args = mtool.parse_arg()
f = open('../result/{}/{}/{}'.format(args.dataset_name, args.save_path, args.save_name), 'r')

results = []
result = []
run_state = 'START'
for line in f.readlines():
    if '[Real Runing]' in line:
        run_state = 'REAL'
    elif '[Evaluation Runing]' in line:
        run_state = 'EVALUATION'
    elif '[END]' in line:
        run_state = 'END'
    
    if run_state == 'REAL':
        if '/ {' in line:
            method_num = re.findall(r'(\d+)/', line)[0]
            method_param = re.findall(r'{([\s|\S]+)}', line)[0]
            result = [method_num, method_param]
        elif '- Accuracy  :' in line:
            res_accs = re.findall(r'\d+.\d*', line)
            result += [float(i) for i in res_accs]
        elif '- Train Time   :' in line:
            train_time = re.findall(r'\d+.\d*', line)
            result += [float(i) for i in train_time]
            results.append(result)
    
    elif run_state == 'EVALUATION':
        if '<' in line:
            method_num = re.findall(r'<(\d+)>', line)[0]
            method_name = re.findall(r'> (.*)', line)[0]
            result = ['---', '---', method_num, method_name]
        elif '- Init Time: ' in line:
            init_time = re.findall(r': (.*)', line)[0]
            result += [float(init_time), '---', '---']
            results.append(result)

        elif '/ {' in line:
            method_num = re.findall(r'(\d+)/', line)[0]
            method_param = re.findall(r'{([\s|\S]+)}', line)[0]
            result = [method_num, method_param]
        elif '- CV scores:' in line:
            cv_scores = re.findall(r'\d+.\d*', line)
            result += [float(i) for i in cv_scores]
        elif '- CV Time   :' in line:
            cv_time = re.findall(r': (.*)', line)[0]
            result += [float(cv_time)]
            results.append(result)


pd.DataFrame(results).to_csv('../result/{}/{}/{}'.format(args.dataset_name, args.save_path, args.save_name[:-3])+'csv', index=False)
        
