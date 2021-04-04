import datetime
import json
import os
import random
import time
from copy import deepcopy

import pytz
from colorama import Fore
from seatable_api import Base

tag_choices = [
    'Msgd', 'Madm', 'Mamw',
    'Mcon', 'Mcos', 'Mpla',
    'Asgd', 'Aadm', 'Aamw',
    'Acon', 'Acos', 'Apla',
    'Naug', 'Raug', 'Rfea',
]
# best= {bac} ,  topk= {kac} , ({exp})


class SeatableLogger(object):
    def __init__(self, exp_path):
        # self.coop = [
        #     # '62b3e51771db4e3998bd6b8df50e8357@auth.local',
        #     # 'f501db34b2a54f6397f39de24cac51dc@auth.local',
        #     # '9bfdb13c6e20495487bb554ce9ccef45@auth.local',
        # ]
        # ssl_aug_api_token = '3240b3ef535e92da60150c6748e87c3e355ff7ea'
        # server_url = 'https://cloud.seatable.cn'
        # base = Base(ssl_aug_api_token, server_url)
        # base.auth()
        # self.base = base
        # self.rid = None
        self.dd = {}
        self.exp_path = exp_path
        self.last_t = time.time()
    
    def create_or_upd_row(self, table_name, vital=False, **kwargs):
        if not vital and (time.time() - self.last_t < 10):
            return
        self.last_t = time.time()
        
        tags = []
        new_kw = deepcopy(kwargs)
        for k, v in kwargs.items():
            if k in tag_choices:
                if v:
                    tags.append(k)
                new_kw.pop(k)
        
        kwargs = new_kw
        dd = dict(**kwargs)
        if len(tags) > 0:
            dd['tags'] = tags
        dd['last_upd'] = datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
        dd['exp_path'] = self.exp_path
        exp_dirname, datetime_dirname = self.exp_path.split(os.path.sep)[-2:]
        dd['exp'] = exp_dirname
        
        if dd.get('pr', 0) == -1 and 'pr' in self.dd:
            dd['pr'] = -self.dd['pr']
        self.dd.update(dd)
        with open(os.path.join(self.exp_path, 'seatable.json'), 'w') as fp:
            json.dump({'table_name': table_name, 'dd': self.dd}, fp)
        
        # if self.rid is None:
        #     # dd['coop'] = self.coop
        #     self.rid = self.base.append_row(table_name, dd)['_id']
        #     SeatableLogger.logging(Fore.LIGHTGREEN_EX, 'created')
        # else:
        #     try:
        #         self.base.update_row(table_name, self.rid, dd)
        #         if random.randrange(16) == 0:
        #             SeatableLogger.logging(Fore.LIGHTBLUE_EX, 'updated')
        #     except ConnectionError:
        #         # dd['coop'] = self.coop
        #         self.rid = self.base.append_row(table_name, dd)['_id']
        #         SeatableLogger.logging(Fore.RED, 're-created')

    @staticmethod
    def logging(clr, msg):
        print(clr + f'[seatable] {msg}')
