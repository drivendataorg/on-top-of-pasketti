import os, sys, logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict
import subprocess
import torch
PROJ='csrw'

#sys.path.insert(0, PROJ)



def get_model_names():
    model_names = defaultdict(list)
    for fpath in glob(f"../data/*KF*"):
        model_name = os.path.basename(fpath)
        m = model_name.split("_KF")[0]
        model_names[m].append(model_name)
    for k, vs in model_names.items():
        model_names[k] = sorted(vs)
    return model_names

def load_model_preds(model_names, model_weights):
    for i, (k, vs) in enumerate(model_names.items()):
        for j, v in enumerate(vs):
            pred = util.load_dump(f"../data/{v}/pred_test.dump").sort_values('uid').reset_index()
            if j==0:
                preds = pred
                preds['pred'] = preds.pred/len(vs)
            else:
                preds['pred'] = preds.pred + pred.pred/len(vs)
        if i==0:
            all_preds = preds
            all_preds['pred'] = all_preds.pred.apply(lambda x: x*model_weights[k])
        else:
            all_preds['pred'] = all_preds.pred + preds.pred.apply(lambda x: x*model_weights[k])
    return all_preds


def main():



    model_names = get_model_names()
    os.system('ls -ltrh ../data/')
    os.system(f'ls -ltrh ../data/{PROJ}/')
    os.system('ls -ltrh ../')
    os.system('ls -ltrh')
    sub = pd.DataFrame(util.load_json_lines('../data/csrw/submission_format.jsonl'))

    if next(iter(model_names)).startswith('CSRW'):
        vbs = 124
        max_gen_len = 144
        temp = 1
        n_beam = 2
        dynamic_beam = 0
        #if len(sub)==9000:
        #    max_gen_len = 4
        test_ds_cls = 'QWASRDynamicDataset'
        test_ds_cls = 'QWASRIterDataset'
        use_flash2 = ''
        max_sec = ""
    else:
        vbs = 64
        max_gen_len = 144
        temp = 1
        n_beam = 5
        dynamic_beam = 0
        #test_ds_cls = 'QWASRDynamicDataset'
        test_ds_cls = 'QWASRIterDataset'
        use_flash2 = '-use_flash2'
        use_flash2 = ''
        max_sec = "-max_sec 30"
        max_sec = ""


    if torch.cuda.is_available():
        torch_dtype = 'bfloat16'
    else:
        torch_dtype = 'float32'
        max_gen_len = 1
    #os.system(f'PYTHONPATH="../:../pkg" python -c "from flash_attn import flash_attn_func; print("flash_attn_func")')
    for model_name, vs in model_names.items():
        kfids = " ".join([v.split("_KF")[-1] for v in vs])
        print(model_name, kfids)
        cmd = f'PYTHONPATH="../:../pkg" python sft.py -kn {len(vs)} -kfids "{kfids}" -m predict -data_type test -ds {model_name[:4].lower()} -vbs {vbs} -model_name {model_name} -n_dl_worker 12 \
        -torch_dtype {torch_dtype} -restore -do_test -temp {temp} -n_beam {n_beam} -dynamic_beam {dynamic_beam} -max_gen_len {max_gen_len} -test_ds_cls {test_ds_cls} {max_sec} -seed 1000 {use_flash2}'
        print(cmd)
        os.system(cmd)

    #preds = load_model_preds(model_names)
    fpaths = glob(f"../data/*KF*")
    assert len(fpaths)==1
    preds = util.load_dump(f"{fpaths[0]}/pred_test.dump")
    if model_name.startswith('CSRW'):
        del sub['orthographic_text']
        preds = preds.rename(columns={"pred":"orthographic_text"})
    else:
        del sub['phonetic_text']
        preds = preds.rename(columns={"pred": "phonetic_text"})
    preds = sub.merge(preds, on="utterance_id")
    preds.to_json('../../../submission/submission.jsonl', orient='records', lines=True)

if __name__ == "__main__":
    # workdir
    os.system('ls')

    # prepare
    os.system(f'mkdir -p {PROJ}/data/{PROJ} && cd {PROJ}/data/{PROJ} && ln -s ../../../../data/submission_format.jsonl submission_format.jsonl')
    os.system(f'mkdir -p {PROJ}/data/{PROJ} && cd {PROJ}/data/{PROJ} && ln -s ../../../../data/utterance_metadata.jsonl utterance_metadata.jsonl')
    os.system(f'mkdir -p {PROJ}/data/{PROJ} && cd {PROJ}/data/{PROJ} && ln -s ../../../../data/audio audio')

    # code dir
    os.chdir(f'{PROJ}/{PROJ}')
    sys.path.insert(0, f".")
    sys.path.insert(0, f"../")
    import util
    main()
