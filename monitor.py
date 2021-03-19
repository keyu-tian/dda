import json
import os
import random
import sys
import time

import colorama
from seatable_api import Base


def terminate(terminate_file):
    if os.path.exists(terminate_file):
        os.remove(terminate_file)
        print(colorama.Fore.CYAN + '[monitor] terminated.')
        exit(-1)


def create_or_upd_explore_table(base, table_name, rid, dd):
    if rid is None:
        q = base.filter(table_name, f"exp_path='{dd['exp_path']}'")
        if q.exists():
            q.update(dd)
            return q.get()['_id'], False
        else:
            return base.append_row(table_name, dd)['_id'], True
    else:
        try:
            base.update_row(table_name, rid, dd)
            ret = rid, False
        except ConnectionError:
            ret = base.append_row(table_name, dd)['_id'], True
        return ret


def main():
    colorama.init(autoreset=True)
    ssl_aug_api_token = '3240b3ef535e92da60150c6748e87c3e355ff7ea'
    server_url = 'https://cloud.seatable.cn'
    base = Base(ssl_aug_api_token, server_url)
    base.auth()
    
    exp_dir_name = sys.argv[1]
    exp_path = os.path.join(os.getcwd(), exp_dir_name)
    seatable_file = os.path.join(exp_path, 'seatable.json')
    terminate_file = f'{exp_path}.terminate'
    
    while not os.path.exists(seatable_file):
        time.sleep(20)
        terminate(terminate_file)
    
    with open(seatable_file, 'r') as fp:
        last_dd = json.load(fp)
    
    first = True
    rid = None
    while True:
        time.sleep(10)
        attempts, max_att = 0, 5
        while attempts < max_att:
            try:
                with open(seatable_file, 'r') as fp:
                    data = json.load(fp)
                    table_name, dd = data['table_name'], data['dd']
            except json.decoder.JSONDecodeError:
                attempts += 1
            else:
                break
        if attempts == max_att:
            raise json.decoder.JSONDecodeError
        
        final = dd['pr'] > 0.99999
        if not final and not first and dd == last_dd:
            terminate(terminate_file)
            continue
            
        first = False
        last_dd = dd
        attempts, max_att = 0, 5
        while attempts < max_att:
            try:
                rid, created = create_or_upd_explore_table(base, table_name, rid, dd)
            except Exception as e:
                attempts += 1
            else:
                break
        if attempts == max_att:
            raise e
        
        logging = random.randrange(16) == 0
        logging |= created
        if logging:
            print(colorama.Fore.LIGHTBLUE_EX + f'[monitor] {"created" if created else "updated"}')
        terminate(terminate_file)


if __name__ == '__main__':
    main()
