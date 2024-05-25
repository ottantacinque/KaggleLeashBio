from pathlib import Path

def find_latest_ckpt_path(fold, dir_path):
    ckpt_list = list(dir_path.glob(f'fold_{fold}*.ckpt'))

    latest_ckpt = max(ckpt_list, key=lambda p: p.stat().st_mtime)
    print(f'Latest checkpoint file: {latest_ckpt}')
    return latest_ckpt


def del_old_ckpt_path(fold, dir_path):
    # Assuming paths.CKPT_ROOT and fold_num are already defined
    ckpt_list = list(dir_path.glob(f'fold_{fold}*.ckpt'))
    print(f'find {len(ckpt_list)} ckpts')
    print(ckpt_list)

    # Find the latest checkpoint file
    if ckpt_list:
        latest_ckpt = max(ckpt_list, key=lambda p: p.stat().st_mtime)
        print(f'Latest checkpoint file: {latest_ckpt}')

        # Delete all other checkpoint files
        for ckpt in ckpt_list:
            if ckpt != latest_ckpt:
                print(f'Deleting checkpoint file: {ckpt}')
                ckpt.unlink()
    else:
        print('No checkpoint files found')