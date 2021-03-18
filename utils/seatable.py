import random
from copy import deepcopy

import os

from colorama import Fore
import pytz
import datetime
from seatable_api import Base

tag_choices = [
    'Msgd', 'Madm', 'Mamw',
    'Mcon', 'Mcos', 'Mpla',
    'Asgd', 'Aadm', 'Aamw',
    'Acon', 'Acos', 'Apla',
]


class SeatableLogger(object):
    def __init__(self, exp_path):
        self.coop = ['62b3e51771db4e3998bd6b8df50e8357@auth.local', 'f501db34b2a54f6397f39de24cac51dc@auth.local']
        ssl_aug_api_token = 'f29f99601183940676df44d9b9d253499fdd7eb1'
        server_url = 'https://cloud.seatable.cn'
        base = Base(ssl_aug_api_token, server_url)
        base.auth()
        self.base = base
        self.rid = None
        self.exp_path = exp_path
    
    def create_or_upd_row(self, table_name, **kwargs):
        tags = []
        new_kw = deepcopy(kwargs)
        for k, v in kwargs.items():
            if k in tag_choices and v:
                tags.append(k)
                new_kw.pop(k)
        
        kwargs = new_kw
        dd = dict(tags=tags, **kwargs)
        dd['last_upd'] = datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
        dd['exp_path'] = self.exp_path
        exp_dirname, datetime_dirname = self.exp_path.split(os.path.sep)[-2:]
        dd['exp'] = exp_dirname
        
        if self.rid is None:
            dd['coop'] = self.coop
            self.rid = self.base.append_row(table_name, dd)['_id']
            SeatableLogger.logging(Fore.LIGHTGREEN_EX, 'created')
        else:
            try:
                self.base.update_row(table_name, self.rid, dd)
                if random.randrange(16) == 0:
                    SeatableLogger.logging(Fore.LIGHTBLUE_EX, 'updated')
            except ConnectionError:
                dd['coop'] = self.coop
                self.rid = self.base.append_row(table_name, dd)['_id']
                SeatableLogger.logging(Fore.RED, 're-created')

    @staticmethod
    def logging(clr, msg):
        print(clr + f'[seatable] {msg}')
