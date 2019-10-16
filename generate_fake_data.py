import pandas as pd
import copy
import re
import numpy as np

app_map = [(re.compile(r'cootek\.smartinput\.(international|mainland)\.(ios|android).*'), 'keyboard')]
plugin_map = [(re.compile(r'(cootek\.smartinput\.android|com\.cootek\.smartinputv5)\.skin\..*'), 'skin'),
              (re.compile(r'(cootek\.smartinput.android|com\.cootek\.smartinputv5)\.language.*'), 'language'),
              (re.compile(r'(cootek\.smartinput\.android|com\.cootek\.smartinputv5)\.font.*'), 'font'),
              (re.compile(r'(cootek\.smartinput\.android|com\.cootek\.smartinputv5)\.emoji.*'), 'emoji'),
              (re.compile(r'cootek.smartinput.android.*touchpal.emoji.*'), 'emoji'),
              (re.compile(r'(cootek\.smartinput\.android|com\.cootek\.smartinputv5)\.sticker.*'), 'sticker'),
              (re.compile(r'(cootek\.smartinput\.android|com\.cootek\.smartinputv5)\.celldict.*'), 'celldict'),
              (re.compile(r'com.cootek.smartinputv5.boomtextv2.*'), 'boomtext')]
matrix_plugin_map = [
    (re.compile(r'com\.color\.call\.flash\.colorphone\.theme\..*'), 'com.color.call.flash.colorphone.theme')]
regex_map = app_map + plugin_map + matrix_plugin_map


def app_name2bundle(app_name):
    if not app_name:
        return app_name
    for (k, v) in regex_map:
        if k.search(app_name):
            return v
    return app_name


with open('20190601.json') as f:
    forecast = pd.read_json(f, lines=True)
# forecast['app_name'] = forecast['app_name'].map(app_name2bundle)
# forecast = forecast['impression'].groupby \
#         (by=[forecast.app_name, forecast.country, forecast.id_type, forecast.platform, forecast.tu]).sum().reset_index(drop=True)
# forecast = forecast.replace(['none', ''], np.nan).dropna().reset_index(drop=True)
fake_real = copy.deepcopy(forecast)
mask = pd.Series(fake_real['platform'] == 'sniper') | pd.Series(fake_real['platform'] == 'flurry')
fake_real = fake_real.join(pd.DataFrame({'mask': mask}))
fake_real.loc[fake_real['mask'] == True, 'impression'] = \
    fake_real.loc[fake_real['mask'] == True, 'impression'] / 2
fake_real = fake_real.reset_index(drop=True)
fake_real.drop(columns=['mask'], inplace=True)
# print(
#     forecast['impression'].groupby(forecast.platform).sum() - fake_real['impression'].groupby(fake_real.platform).sum())
fake_real.to_json('fake_real.json', orient='records', lines=True)
