import os
import subprocess
import sys
import re

def generate_funcs(so_path, output_dir, base_addr=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing {so_path} with base address {hex(base_addr)}...")
    # Run objdump
    result = subprocess.run(['objdump', '-d', so_path], capture_output=True, text=True)
    lines = result.stdout.split('\n')
    
    current_func_name = None
    current_func_lines = []
    func_index = 0
    
    for line in lines:
        # Match function header: 0000000000002120 <tvmgen_default_fused_add_layout_transform>:
        match = re.match(r'^([0-9a-f]+) <([^>]+)>:', line)
        if match:
            if current_func_name and current_func_lines:
                # Save previous function
                save_func(output_dir, func_index, current_func_name, current_func_lines)
                func_index += 1
            
            current_func_name = match.group(2)
            current_func_lines = []
        elif current_func_name:
            # Match instruction line:    2120:       55                      push   %rbp
            match_ins = re.match(r'^\s+([0-9a-f]+):', line)
            if match_ins:
                rel_addr = int(match_ins.group(1), 16)
                abs_addr = base_addr + rel_addr
                # Format to match get_func_range: "address: instruction"
                clean_line = f"{hex(abs_addr)[2:]}: {line.split(':', 1)[1].strip()}"
                current_func_lines.append(clean_line)
    
    # Save last function
    if current_func_name and current_func_lines:
        save_func(output_dir, func_index, current_func_name, current_func_lines)
    
    print(f"Generated {func_index} function files in {output_dir}")

def save_func(output_dir, index, name, lines):
    # Only save TVM generated functions
    if not name.startswith('tvmgen_default_fused'):
        return
        
    filename = f"{index:04d}.{name}.txt"
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write('\n'.join(lines))

if __name__ == "__main__":
    base_addr = int(sys.argv[3], 16) if len(sys.argv) > 3 else 0
    generate_funcs(sys.argv[1], sys.argv[2], base_addr)
