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
    "logging.project_name=smac_whisper_sweep",
    "training.num_epochs=12",
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

    config_dict = dict(config)

    # 2. Extract and remove the helper variables safely using pop()
    bb_w = config_dict.pop("training.bb_warmup")
    bb_h = config_dict.pop("training.bb_hold")
    head_w = config_dict.pop("training.head_warmup")
    head_h = config_dict.pop("training.head_hold")

    # 3. Calculate the decay phases safely
    if bb_w + bb_h >= 1.0:
        bb_h = 0.95 - bb_w # Force a small decay phase
    if head_w + head_h >= 1.0:
        head_h = 0.95 - head_w
        
    bb_decay = 1.0 - bb_w - bb_h
    head_decay = 1.0 - head_w - head_h

    # 4. Format the lists as strings without spaces for Hydra/CLI compatibility
    backbone_phase_ratio = f"[{bb_w:.4f},{bb_h:.4f},{bb_decay:.4f}]"
    head_phase_ratio = f"[{head_w:.4f},{head_h:.4f},{head_decay:.4f}]"

    # 5. Add the new properly formatted keys to the dictionary
    config_dict["scheduler.backbone.phase_ratio"] = backbone_phase_ratio
    config_dict["scheduler.head.phase_ratio"] = head_phase_ratio

    for key, value in config_dict.items():
        cmd.append(f"{key}={value}")

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
    dask_client = Client("tcp://192.168.14.13:8786")
    
    print(f"Connected to Dask Cluster: {dask_client.dashboard_link}")
    cs = ConfigurationSpace()
    
    # # class_dropout = Float('model.classifier_dropout', (0.01, 0.4), default=0.1)
    # lora_r = Categorical('model.lora_r', [8,16,24], default=16)
    # lora_alpha = Categorical('model.lora_alpha', [16,24,32,64], default=16)
    # lora_dropout = Float('model.lora_dropout', (0.01, 0.2), default=0.1)

 
    # whisper_variant = Categorical("whisper", ["tiny", "small", "base", "medium"], default="small")

    gamma = Float("loss.gamma", (0.1, 0.6), log=False, default=0.5)
    max_grad_norm = Categorical("training.max_grad_norm", [1.0, 5.0, 10.0], ordered=True, default=1.0)


    # --- Training ---
    backbone_lr = Float("training.backbone_lr", (1e-7, 5e-5), log=True, default=8e-6)
    head_lr = Float("training.head_lr", (5e-5, 5e-3), log=True, default=2e-4)
    wd = Float("training.weight_decay", (1e-5, 5e-2), log=True, default=1e-2)

    #scheduler params
    bb_warmup = Float("training.bb_warmup", (0.05, 0.15), default=0.1)
    bb_hold = Float("training.bb_hold", (0.1, 0.4), default=0.2)

    # --- Head Scheduler Params ---
    head_warmup = Float("training.head_warmup", (0.0005, 0.05), default=0.05)
    head_hold = Float("training.head_hold", (0.1, 0.4), default=0.2)

    classifier_dropout = Float('model.classifier_dropout', (0.01, 0.4), default=0.15)
    
    background_p = Float("augmentation.background_noise.p", (0.0, 0.5), default=0.1)

    #enable_age_head: false
    # age_head_lambda: 0.1

    age_head = Categorical("model.enable_age_head", [True, False], default=False)
    age_lambda = Float("model.age_head_lambda", (0.05, 0.4), default=0.1)

    con_lambda = EqualsCondition(age_lambda, age_head, True)


    #Base hyperparameters
    cs.add([
        backbone_lr, head_lr, wd, gamma, max_grad_norm, bb_warmup, bb_hold, 
        head_warmup, head_hold, classifier_dropout, background_p, age_head, age_lambda, con_lambda
    ])

    # 2. Define Scenario
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = f"whisper_sweep_{timestamp}"
    run_name = "whisper_sweep_2026-03-31_21-20-41"

    scenario = Scenario(
        cs,
        name=run_name, # Name of the run
        output_directory=Path("smac3_output"), # Top-level output directory
        # deterministic=True, # Set to False if your training is stochastic even with fixed seed
        n_trials=200,        # Number of evaluations
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
            n_configs=10, # Number of initial configurations to run
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
