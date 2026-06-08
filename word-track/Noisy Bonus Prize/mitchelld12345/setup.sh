#!/bin/bash
set -eou pipefail

micromamba create -n child_asr python=3.12 -y
micromamba run -n child_asr pip install -r requirements.txt
