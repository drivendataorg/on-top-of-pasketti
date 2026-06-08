import logging
import subprocess
import re
import sys
import socket
from pathlib import Path
from datetime import datetime
import numpy as np
from ConfigSpace import (
    Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, 
    EqualsCondition, ForbiddenEqualsClause, ForbiddenAndConjunction,
    InCondition, 
    AndConjunction, OrConjunction
)
from smac import HyperparameterOptimizationFacade, Scenario
from dask.distributed import Client

# --- Configuration ---
BASE_COMMAND = ["uv", "run", "src/run_pipeline.py", "--config-name=default"]

FIXED_ARGS = [
    "model=whisper_ctc",
    "training=whisper-base",
    "logging.use_wandb=true",
]

def to_hydra_val(val):
    """Helper to convert python types to hydra/yaml string representations."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        # Wrap strings in quotes if they contain spaces or special chars
        # Added more special chars to be safe: / -
        if any(c in val for c in " [](),/-"):
            return f"'{val}'"
        return val
    return str(val)

def train_model(config: Configuration, seed: int = 0) -> tuple[float, dict]:
    hostname = socket.gethostname()
    cmd = BASE_COMMAND.copy()
    cmd.extend(FIXED_ARGS)
    for key, value in config.items():
        cmd.append(f"{key}={to_hydra_val(value)}")

    # --- 2. Handle Pretraining Logic ---

    print(f"\n[SMAC] Running on {hostname}: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        match = re.search(r"Cross Validation Complete! Mean Val PER:\s*([0-9.]+)", output)
             
        if match:
            score = float(match.group(1))
            print(f"[SMAC] Run finished on {hostname}. Score: {score}")
            return score, {"hostname": hostname}
        else:
            print(f"[SMAC] Could not find score in output on {hostname}!")
            return 1.0, {"hostname": hostname}

    except subprocess.CalledProcessError as e:
        print(f"[SMAC] Training failed on {hostname} with error code {e.returncode}")
        print(e.stderr)
        return 1.0, {"hostname": hostname}
    

def main():
    # --- 1. Setup Distributed Context ---
    # Connect to the Dask Scheduler. 
    # Replace '192.168.1.100' with the IP of the computer running the scheduler.
    # If running locally for testing, Client() without args creates a local cluster.
    dask_client = Client("tcp://192.168.14.28:8788")
    
    print(f"Connected to Dask Cluster: {dask_client.dashboard_link}")
    cs = ConfigurationSpace()
    
    # # class_dropout = Float('model.classifier_dropout', (0.01, 0.4), default=0.1)
    # lora_r = Categorical('model.lora_r', [8,16,24], default=16)
    # lora_alpha = Categorical('model.lora_alpha', [16,24,32,64], default=16)
    # lora_dropout = Float('model.lora_dropout', (0.01, 0.2), default=0.1)

 
    whisper_variant = Categorical("whisper", ["tiny", "small", "base", "medium"], default="small")

    # --- Training ---
    backbone_lr = Float("training.backbone_lr", (1e-7, 5e-5), log=True, default=8e-6)
    head_lr = Float("training.head_lr", (1e-5, 1e-3), log=True, default=2e-4)
    wd = Float("training.weight_decay", (1e-7, 1e-2), log=True, default=1e-2)


    #Base hyperparameters
    cs.add([
        whisper_variant, backbone_lr, head_lr, wd
    ])

    # 2. Define Scenario
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"dino_frozen_sweep_{timestamp}"
    # run_name = "dino_tiled_sweep_2025-12-12_16-46-11"

    scenario = Scenario(
        cs,
        name=run_name, # Name of the run
        output_directory=Path("smac3_output"), # Top-level output directory
        # deterministic=True, # Set to False if your training is stochastic even with fixed seed
        n_trials=50,        # Number of evaluations
        #walltime_limit=3600 * 9, # 15 hours
        n_workers=20 # 6 gpu's available
    )

    # 3. Create SMAC Object
    smac = HyperparameterOptimizationFacade(
        scenario,
        train_model,
        dask_client=dask_client,
        overwrite=False, # Overwrite previous run results
        initial_design=HyperparameterOptimizationFacade.get_initial_design(
            scenario,
            n_configs=5, # Number of initial configurations to run
        ),
        intensifier=HyperparameterOptimizationFacade.get_intensifier(
            scenario,
            max_config_calls=1, # Disable intensification (evaluate each config once)
        ),
    )

    # 4. Run Optimization
    print("Starting SMAC Optimization...")
    incumbent = smac.optimize()

    # 5. Report Results
    print("\nOptimization finished!")
    print(f"Best Configuration found:\n{incumbent}")
    
    # Validate the best config (optional, just runs it one last time)
    # cost = train_model(incumbent)
    # print(f"Best Score: {1.0 - cost}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
