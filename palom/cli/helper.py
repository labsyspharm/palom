import argparse
import pathlib
import sys
import yaml


def main(argv=sys.argv):

    parser = argparse.ArgumentParser(
        description=(
            'Generate configuration yaml file for palom-svs run'
        )
    )
    
    parser.add_argument(
        '-i',
        metavar='input-dir',
        help='directory that contains all the SVS files to be processed',
        required=True
    )
    parser.add_argument(
        '-n',
        metavar='ref-name-pattern',
        help='filename pattern of the reference SVS file',
        required=True
    )
    parser.add_argument(
        '-o',
        metavar='output-ome-path',
        help='full path to the output ome-tiff file',
        required=True
    )
    parser.add_argument(
        '-c',
        metavar='yaml-path',
        help='full path to the resulting configuration yaml file',
        required=True
    )
    
    args = parser.parse_args(argv[1:])
    
    if len(argv) == 1:
        parser.print_help()
        return 0

    svs_dir = pathlib.Path(args.i)
    name_pattern = args.n
    output_path = pathlib.Path(args.o)
    yaml_path = pathlib.Path(args.c)

    svs_paths = sorted(svs_dir.glob('*.svs'))
    svs_paths.sort(key=lambda x: x.name.split('_')[-2])

    assert len(svs_paths) > 0
    assert '.ome.tif' in str(output_path)
    assert '.yml' in str(yaml_path) or '.yaml' in str(yaml_path)

    ref_slide = None
    for svs in svs_paths:
        if svs.match(name_pattern):
            print('Reference file found {}'.format(svs.name))
            ref_slide = svs
            svs_paths.remove(svs)
            break

    assert ref_slide is not None, (
        f"Cannot find reference SVS file with name pattern: {name_pattern}"
    )

    channel_names = [
        '-'.join(p.name.split('_')[-2:][::-1]).replace('.svs', '') 
        for p in ([ref_slide] + svs_paths)
    ]

    config = {
        'input dir': str(svs_dir),
        'output full path': str(output_path),
        'reference image': {
            'filename': str(ref_slide).replace(str(svs_dir), '.'),
            'output mode': 'hematoxylin',
            'channel name': channel_names[0]
        },
        'moving images': [
            {
                'filename': str(s).replace(str(svs_dir), '.'),
                'output mode': 'aec' if not s.match(name_pattern) else 'hematoxylin',
                'channel name': n
            }
            for s, n in zip(svs_paths, channel_names[1:])
        ]
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


if __name__ == '__main__':
    sys.exit(main())
