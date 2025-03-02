import json
import glob
from gec.utils.postprocess import remove_pnx, pnx_tokenize, space_clean
from camel_tools.utils.dediac import dediac_ar


def read_txt(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]

def read_preds(preds_dir):
    preds = []
    files = glob.glob(f'{preds_dir}/*json')
    for i in range(len(files)):
        file = f'{preds_dir}/{i}.json'
        with open(file) as f:
            pred = json.load(f)
        preds.append(pred['output'])
    return preds

def write_preds(preds, path, preproc=False, delete_pnx=False, clean_space=False):
    preds = [clean_txt(pred) for pred in preds]
    if clean_space:
        preds = space_clean(preds)

    if preproc:
        preds = pnx_tokenize(preds)

    if delete_pnx:
        preds = remove_pnx(preds)

    with open(path, mode='w') as f:
        for pred in preds:
            f.write(pred.strip())
            f.write('\n')    

def clean_txt(txt):
    txt = txt.replace('<output>', '').replace('</output>', '')
    txt = txt.replace('<input>', '').replace('</input>', '')
    txt = txt.replace('\n', '')
    txt = dediac_ar(txt)
    txt = txt.strip()
    return txt


if __name__ == '__main__':
    # Dev sets
    for dir in ['jais-outputs', 'fanar-outputs-new', 'gpt4o-outputs', 'gpt3.5-outputs']:
        for exp in ['zero-shot-ar', 'few-shot-ar']:
            for dataset in ['qalb14', 'zaebuc', 'madar']:
                pred_path = f'/scratch/ba63/arabic-text-editing/llms-outputs/{dir}/{exp}/{dataset}/dev'
                print(f'Processing Data in {pred_path}')
                preds = read_preds(pred_path)
                output_path=f'/scratch/ba63/arabic-text-editing/llms-outputs/{dir}/{exp}/{dataset}'
                
                print(f'Writing Data to {output_path}')
                if dataset == 'madar':
                    write_preds(path=f'{output_path}/dev.preds.txt', preds=preds,
                                preproc=False, delete_pnx=False, clean_space=True)
                else:
                    write_preds(path=f'{output_path}/dev.preds.txt', preds=preds, preproc=True)
                    write_preds(path=f'{output_path}/dev.preds.nopnx.txt', preds=preds, preproc=True, delete_pnx=True)
    

    # Test sets
    for dataset in ['qalb14', 'qalb15', 'zaebuc', 'madar']:
        pred_path = f'/scratch/ba63/arabic-text-editing/llms-outputs/gpt4o-outputs/few-shot-en/{dataset}/test'
        print(f'Processing Data in {pred_path}')
        preds = read_preds(pred_path)
        output_path=f'/scratch/ba63/arabic-text-editing/llms-outputs/gpt4o-outputs/few-shot-en/{dataset}'
        
        print(f'Writing Data to {output_path}')
        if dataset == 'madar':
            write_preds(path=f'{output_path}/test.preds.txt', preds=preds,
                        preproc=False, delete_pnx=False, clean_space=True)
        else:
            write_preds(path=f'{output_path}/test.preds.txt', preds=preds, preproc=True)
            write_preds(path=f'{output_path}/test.preds.nopnx.txt', preds=preds, preproc=True, delete_pnx=True)

