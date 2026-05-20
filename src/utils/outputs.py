import os 
from datetime import datetime

def make_output_dir(config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = config["logging"]["run_name"]
    
    out_dir = f"{config['logging']['base_dir']}/{timestamp}_{name}"
    os.makedirs(out_dir, exist_ok=True)
    
    return out_dir