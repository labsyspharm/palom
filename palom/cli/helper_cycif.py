import argparse
import pathlib
import sys
import yaml


def main(argv=sys.argv):

    parser = argparse.ArgumentParser(
        description=(
            'Generate configuration yaml file for palom-cycif run'
        )
    )
    
    parser.add_argument(
        '-i',
        metavar='input-dir',
        help='directory that contains all the TIF files to be processed',
        required=True
    )
    parser.add_argument(
        '-n',
        metavar='ref-name-pattern',
        help='filename pattern of the reference TIF file',
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

    cycif_dir = pathlib.Path(args.i)
    name_pattern = args.n
    output_path = pathlib.Path(args.o)
    yaml_path = pathlib.Path(args.c)

    cycif_paths = sorted(svs_dir.glob('*.tif'))
    cycif_paths.sort(key=lambda x: x.name.split('_')[-2])

    assert len(cycif_paths) > 0
    assert '.ome.tif' in str(output_path)
    assert '.yml' in str(yaml_path) or '.yaml' in str(yaml_path)

    ref_slide = None
    for tif in cycif_paths:
        if tif.match(name_pattern):
            print('Reference file found {}'.format(tif.name))
            ref_slide = tif
            cycif_paths.remove(tif)
            break

    assert ref_slide is not None, (
        f"Cannot find reference TIF file with name pattern: {name_pattern}"
    )

    channel_names = [
        '-'.join(p.name.split('_')[-2:][::-1]).replace('.tif', '') 
        for p in ([ref_slide] + cycif_paths)
    ]

    config = {
        'input dir': str(cycif_dir),
        'output full path': str(output_path),
        'reference image': {
            'filename': str(ref_slide).replace(str(cycif_dir), '.'),
            'output mode': 'multichannel',
            'channel names': 'DAPI, FITC, Cy3, Cy5'
        },
        'moving images': [
            {
                'filename': str(s).replace(str(cycif_dir), '.'),
                'output mode': 'multichannel'
                'channel names': "DAPI, FITC, Cy3, Cy5"
            }
            for s, n in zip(cycif_paths, channel_names[1:])
        ]
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


if __name__ == '__main__':
    sys.exit(main())
