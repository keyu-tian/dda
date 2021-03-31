import os


def dfs(cwd, upd_fn):
    file_names = os.listdir(cwd)
    for name in file_names:
        path = os.path.join(cwd, name)
        if os.path.isdir(path):
            dfs(path, upd_fn)
        else:
            # if name == 'cfg.yaml' or name.endswith('.sh'):
            if name == 'run.sh':
                upd_fn(path)


if __name__ == '__main__':
    def upd(path):
        with open(path, 'r', encoding='utf-8') as fp:
            ctt = fp.read()
        with open(path, 'w', encoding='utf-8') as fp:
            fp.write(
                ctt.replace(
                    '# # touch "${EXP_DIR}".terminate',
                    '# touch "${EXP_DIR}".terminate',
                )
            )
    
    dfs(os.getcwd(), upd)
    

