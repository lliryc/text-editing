import os
import argparse
import glob


def get_best_checkpoint_gec(model_path, m2_file_name, add_nopnx_eval):
    checkpoints = glob.glob(os.path.join(model_path, 'checkpoint-*/'))
    checkpoints += [model_path]
    checkpoint_scores = []

    for checkpoint in checkpoints:
        # m2_files = glob.glob(os.path.join(checkpoint, '*.m2'))
        # m2_files = [f for f in m2_files if not f.endswith('.nopnx.m2')]

        m2_files = glob.glob(os.path.join(checkpoint, m2_file_name))

        for m2_file in m2_files:
            with open(m2_file) as f:
                lines = f.readlines()
                p = lines[0].strip().split()[-1]
                r = lines[1].strip().split()[-1]
                f1 = lines[2].strip().split()[-1]
                f_05 = lines[3].strip().split()[-1]
                checkpoint_scores.append((m2_file, {'p': p, 'r': r, 'f1': f1, 'f0.5': f_05}))

    best_checkpoint = max(checkpoint_scores, key=lambda x: (x[1]['f0.5'], x[1]['f1']))

    best_checkpoint = {'dir': best_checkpoint[0], 'm2score': best_checkpoint[1]}

    # adding the no pnx eval
    if add_nopnx_eval:
        m2_file_name_nopnx = m2_file_name.replace('.m2', '.nopnx.m2')

        with open(best_checkpoint['dir'].replace(m2_file_name, m2_file_name_nopnx)) as f:
            lines = f.readlines()
            p = lines[0].strip().split()[-1]
            r = lines[1].strip().split()[-1]
            f1 = lines[2].strip().split()[-1]
            f_05 = lines[3].strip().split()[-1]

            best_checkpoint['m2score_nopnx'] = {'p': p, 'r': r, 'f1': f1, 'f0.5': f_05}


    return best_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--m2file')
    parser.add_argument('--add_nopnx_eval')
    args = parser.parse_args()
    best_checkpoint = get_best_checkpoint_gec(model_path=args.model_path,
                                              m2_file_name=args.m2file,
                                              add_nopnx_eval=args.add_nopnx_eval)

    print(best_checkpoint, flush=True)
