#!/usr/bin/env bash

# Use optional first parameter as fallback if parsing fails or ENSEMBLE is false
FALLBACK_DIR="$1"

EXAMPLE_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cp "$EXAMPLE_ROOT_DIR/run_inference.py" "$EXAMPLE_ROOT_DIR/main.py"

WORKING_DIR="$(pwd)"

# Define and create the external directory
EXTERNAL_DIR="$EXAMPLE_ROOT_DIR/external"
mkdir -p "$EXTERNAL_DIR"

export FALLBACK_DIR="$FALLBACK_DIR"
export EXTERNAL_DIR="$EXTERNAL_DIR"
export TEMP_TRACKER="$EXAMPLE_ROOT_DIR/.pack_vars.tmp"
export WORKING_DIR="$WORKING_DIR"
export EXAMPLE_ROOT_DIR="$EXAMPLE_ROOT_DIR"

# --- EXTRACT RUN PATHS AND DOWNLOAD MODELS ---
uv run --with huggingface_hub --with pyyaml python3 -c '
import yaml
import os
import sys
import shutil
import ast
from pathlib import Path
from huggingface_hub import snapshot_download

fallback_dir = os.environ.get("FALLBACK_DIR")
external_dir = os.environ.get("EXTERNAL_DIR")
temp_tracker = os.environ.get("TEMP_TRACKER")

with open("main.py") as f:
    source = f.read()

module = ast.parse(source)
repo_root = Path(os.getcwd()).resolve()
symbols = {
    "PROJECT_ROOT": repo_root,
}


def eval_expr(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in symbols:
        return symbols[node.id]
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "Path" and len(node.args) == 1:
            arg = eval_expr(node.args[0])
            if arg is None:
                return None
            return Path(str(arg))
        return None
    if isinstance(node, ast.BinOp):
        left = eval_expr(node.left)
        right = eval_expr(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Div):
            return Path(str(left)) / str(right)
        if isinstance(node.op, ast.Add):
            return str(left) + str(right)
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for elem in node.elts:
            value = eval_expr(elem)
            if value is None:
                return None
            values.append(value)
        return values
    return None


def normalize_path(value):
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = (repo_root / p)
    return str(p.resolve())


paths_dict = {}
list_paths = []
ensemble = False

for stmt in module.body:
    if not isinstance(stmt, ast.Assign):
        continue
    value = stmt.value
    for target in stmt.targets:
        if not isinstance(target, ast.Name):
            continue

        name = target.id
        evaluated = eval_expr(value)

        if name == "ENSEMBLE":
            if isinstance(evaluated, bool):
                ensemble = evaluated
            continue

        if name in ["PROJECT_ROOT"]:
            if isinstance(evaluated, (str, Path)):
                symbols[name] = Path(str(evaluated))
            continue

        if name in ["OUTPUT_PATH", "OUTPUT_PATH_A", "OUTPUT_PATH_B"]:
            if isinstance(evaluated, (str, Path)):
                paths_dict[name] = normalize_path(evaluated)
            continue

        if name in ["output_paths", "OUTPUT_PATHS"]:
            if isinstance(evaluated, list):
                for item in evaluated:
                    if isinstance(item, (str, Path)):
                        list_paths.append(normalize_path(item))

outputs_to_pack = []
if list_paths:
    outputs_to_pack.extend(list_paths)
elif ensemble:
    if "OUTPUT_PATH_A" in paths_dict:
        outputs_to_pack.append(paths_dict["OUTPUT_PATH_A"])
    if "OUTPUT_PATH_B" in paths_dict:
        outputs_to_pack.append(paths_dict["OUTPUT_PATH_B"])
else:
    if "OUTPUT_PATH" in paths_dict:
        outputs_to_pack.append(paths_dict["OUTPUT_PATH"])

if not outputs_to_pack and fallback_dir:
    outputs_to_pack = [fallback_dir]

if not outputs_to_pack:
    print("Error: Could not extract OUTPUT_PATH(s) from run_inference.py and no argument provided.")
    sys.exit(1)

models_to_pack = set()

# Deduplicate outputs
outputs_to_pack = list(dict.fromkeys(outputs_to_pack))

for out_path in outputs_to_pack:
    config_path = os.path.join(out_path, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "model" in config and "whisper_model_id" in config["model"]:
        repo_id = config["model"]["whisper_model_id"]
    else:
        repo_id = config["model"]["pretrained_name"] 
        
    local_dir_name = repo_id.split("/")[-1]
    local_dir = os.path.join(external_dir, local_dir_name)
    models_to_pack.add(local_dir_name)

    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        print(f"Downloading {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", ".git*", ".cache*"]
        )

    # Scrub metadata/cache to prevent log spam
    for root, dirs, files in os.walk(local_dir):
        for d in [d for d in dirs if d in [".cache", ".git"]]:
            shutil.rmtree(os.path.join(root, d))
        for f in files:
            if f.endswith(".metadata") or f.startswith("."):
                os.remove(os.path.join(root, f))

# Write out the variables for bash to consume
with open(temp_tracker, "w") as f:
    for out_path in outputs_to_pack:
        f.write(f"OUTPUT_PATHS+=(\"{out_path}\")\n")
    for mod in models_to_pack:
        f.write(f"MODEL_DIR_NAMES+=(\"{mod}\")\n")
'

if [ $? -ne 0 ]; then
    rm "$EXAMPLE_ROOT_DIR/main.py"
    rm -f "$TEMP_TRACKER"
    exit 1
fi

OUTPUT_PATHS=()
MODEL_DIR_NAMES=()
source "$TEMP_TRACKER"
rm "$TEMP_TRACKER"

# Pre-zip cleanup of source files
find "$EXAMPLE_ROOT_DIR/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "Zipping specific files only..."
shopt -s extglob
EXCLUDE_1="optimize_decoder"
EXCLUDE_2="length_mark"

ZIP_ARGS=(
    offline_wheels
    src/!($EXCLUDE_1|$EXCLUDE_2)
    main.py
    phoneme_corpus.txt
)

for MODEL_DIR in "${MODEL_DIR_NAMES[@]}"; do
    ZIP_ARGS+=("external/$MODEL_DIR")
done

for output_folder in "${OUTPUT_PATHS[@]}"; do
    RELATIVE_OUTPUT_FOLDER=$(realpath --relative-to="$EXAMPLE_ROOT_DIR" "$output_folder")
    ZIP_ARGS+=("$RELATIVE_OUTPUT_FOLDER/.hydra/config.yaml")

    mapfile -t NEW_MODELS < <(find "$RELATIVE_OUTPUT_FOLDER" -type f \( -name "*best*.pt" -o -name "*best*.pth" \))

    if [ ${#NEW_MODELS[@]} -eq 0 ]; then
        echo "Warning: No files matching *best*.pt or *best*.pth found in $RELATIVE_OUTPUT_FOLDER"
    else
        echo "Found the following model files to pack from $RELATIVE_OUTPUT_FOLDER:"
        printf " - %s\n" "${NEW_MODELS[@]}"
        ZIP_ARGS+=("${NEW_MODELS[@]}")
    fi
done

# --- TARGETED ZIP COMMAND ---
echo "Packing submission with zero compression for speed..."

uv run --with repro-zipfile python3 -c '
import os
import sys
import zipfile
from repro_zipfile import ReproducibleZipFile

working_dir = os.environ.get("WORKING_DIR")
example_root = os.environ.get("EXAMPLE_ROOT_DIR")
zip_name = os.path.join(working_dir, "submission.zip")

# Change to the root directory so relative paths in the zip match expected structure
os.chdir(example_root)

# ZIP_STORED skips the CPU-heavy compression step
with ReproducibleZipFile(zip_name, "w", compression=zipfile.ZIP_STORED) as zp:
    for item in sys.argv[1:]:
        if os.path.isfile(item):
            zp.write(item, arcname=item)
        elif os.path.isdir(item):
            for root, _, files in os.walk(item):
                for file in files:
                    file_path = os.path.join(root, file)
                    zp.write(file_path, arcname=file_path)
        else:
            print(f"Warning: {item} not found or not a valid file/directory.")
' "${ZIP_ARGS[@]}"

rm "$EXAMPLE_ROOT_DIR/main.py"
echo "Submission packed successfully!"
