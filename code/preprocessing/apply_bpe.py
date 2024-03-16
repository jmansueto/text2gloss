import os
import subprocess

def generate_and_apply_bpe_en(input_folder_path, output_folder_path, num_operations=8000):
    """
    Generates BPE codes from the train.en file and applies them to all .en and .asl files in the folder.
    
    Args:
    input_folder_path (str): Path to the folder containing .en and .asl files.
    num_operations (int): Number of BPE merge operations.
    """
    # Define paths
    bpe_codes_path = os.path.join(output_folder_path, 'bpe_codes_en.txt')
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # Generate BPE codes from train.en
    train_en_path = os.path.join(input_folder_path, 'train.en')
    generate_bpe_codes_command = f'subword-nmt learn-bpe -s {num_operations} < {train_en_path} > {bpe_codes_path}'
    subprocess.run(generate_bpe_codes_command, shell=True, check=True)
    
    # Apply BPE to all .en and .asl files in the folder
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.en'):
            input_file_path = os.path.join(input_folder_path, filename)
            output_file_path = os.path.join(output_folder_path, filename)
            apply_bpe_command = f'subword-nmt apply-bpe -c {bpe_codes_path} < {input_file_path} > {output_file_path}'
            subprocess.run(apply_bpe_command, shell=True, check=True)
    
    print("BPE processing for en completed successfully.")

def generate_and_apply_bpe_asl(input_folder_path, output_folder_path, num_operations=8000):
    """
    Generates BPE codes from the train.en file and applies them to all .en and .asl files in the folder.
    
    Args:
    input_folder_path (str): Path to the folder containing .en and .asl files.
    num_operations (int): Number of BPE merge operations.
    """
    # Define paths
    bpe_codes_path = os.path.join(output_folder_path, 'bpe_codes_asl.txt')
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # Generate BPE codes from train.en
    train_en_path = os.path.join(input_folder_path, 'train.asl')
    generate_bpe_codes_command = f'subword-nmt learn-bpe -s {num_operations} < {train_en_path} > {bpe_codes_path}'
    subprocess.run(generate_bpe_codes_command, shell=True, check=True)
    
    # Apply BPE to all .en and .asl files in the folder
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.asl'):
            input_file_path = os.path.join(input_folder_path, filename)
            output_file_path = os.path.join(output_folder_path, filename)
            apply_bpe_command = f'subword-nmt apply-bpe -c {bpe_codes_path} < {input_file_path} > {output_file_path}'
            subprocess.run(apply_bpe_command, shell=True, check=True)
    
    print("BPE processing for asl completed successfully.")

# Example usage:
input_folder_path = '/home/jennamansueto/text2gloss/code/data/phoenix_src_pos/splits'
output_folder_path = '/home/jennamansueto/text2gloss/code/data/phoenix_src_pos/bpe_splits'
generate_and_apply_bpe_en(input_folder_path, output_folder_path)
generate_and_apply_bpe_asl(input_folder_path, output_folder_path)
