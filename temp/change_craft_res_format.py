import os

craft_res_dir = './res_craft_ic15/res_craft_ic15'
files = os.listdir(craft_res_dir)
for file in files:
    with open(os.path.join(craft_res_dir, file), 'r') as f:
        content = f.read()
    lines = content.split('\n')
    new_lines = [line.split(' ')[0] + ',' + line.split(' ')[1] for line in lines if len(line) > 0]
    output = '\n'.join(new_lines)
    with open(os.path.join(craft_res_dir, file), 'w') as f:
        f.write(output)