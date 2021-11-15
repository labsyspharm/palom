import pathlib
import pathlib
import palom.reader

settings = {
    'svs_dir': r'Y:\sorger\data\OHSU\2020OCT-TNP_SARDANA_Phase_1\768473',
    'ref_slide_pattern': '*Hem*',
    'out_dir': r'D:\yc296\20211104-OHSU_SARDANA_P1'
}

svs_dir = pathlib.Path(settings['svs_dir'])

svs_paths = sorted(svs_dir.glob('*.svs'))
svs_paths.sort(key=lambda x: x.name.split('_')[-2])

ref_slide = None
for svs in svs_paths:
    if svs.match(settings['ref_slide_pattern']):
        print('Reference file found {}'.format(svs.name))
        ref_slide = svs
        svs_paths.remove(svs)
        break

channel_names = [
    '-'.join(p.name.split('_')[-2:][::-1]).replace('.svs', '') 
    for p in ([ref_slide] + svs_paths)
]

config = {
    'input dir': str(svs_dir),
    'output full path': '???',
    'reference image': {
        'filename': str(ref_slide).replace(str(svs_dir), '.'),
        'output mode': 'hematoxylin',
        'channel name': channel_names[0]
    },
    'moving images': [
        {
            'filename': str(s).replace(str(svs_dir), '.'),
            'output mode': 'aec',
            'channel name': n
        }
        for s, n in zip(svs_paths, channel_names[1:])
    ]
}

import yaml

with open('zzzz.yml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)
