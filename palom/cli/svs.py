import sys
import argparse
import pathlib
import yamale
from loguru import logger
import datetime
import shutil

from . import schema
from .. import reader, align, write_pyramid
from .. import __version__ as _version

import matplotlib.pyplot as plt


logger.remove()  # All configured handlers are removed
logger.add(
    sys.stderr,
    format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>'
)


@logger.catch
def main(argv=sys.argv):

    parser = argparse.ArgumentParser(
        description=(
            'Align multiple SVS images of the same biospecimen and writes a merged'
            ' pyramidal ome-tiff'
        )
    )
    parser.add_argument(
        '--version', dest='version', default=False, action='store_true',
        help='print version'
    )

    subparsers = parser.add_subparsers(
        title='subcommands',
        dest='subparser_name',
        help=(
            'show configuration yaml example and schema; run using a configuration file'
        )
    )

    parser_show = subparsers.add_parser('show')
    parser_show.add_argument(
        'info-type', choices=['example', 'schema'],
        help='show configuration yaml example/schema in console'
    )
    parser_show.set_defaults(func=show_info)

    parser_run = subparsers.add_parser(
        'run',
        description='process files described in a configuration yaml file'
    )
    parser_run.add_argument(
        '-c', '--config-file',
        metavar='FILE',
        type=argparse.FileType('r'),
        help=(
            'a configuration yaml file; see an example in console by running'
            ' `palom-svs show example`'
        ),
        required=True
    )
    parser_run.set_defaults(func=run)

    args = parser.parse_args(argv[1:])
    
    if len(argv) == 1:
        parser.print_help()
        return 0
    
    if args.version:
        print(f"palom v{_version}")
        return 0

    return args.func(args)


def show_info(args):
    info_type = vars(args)['info-type']

    if info_type == 'example':
        info_path = schema.svs_config_example_path
    else:
        info_path = schema.svs_config_schema_path
    with open(info_path) as f:
        print_text = f.read()
    print(f"\n{info_path}\n\n\n{print_text}\n\n")
    return 0


def run(args):
    config_file = args.config_file
    config_filepath = pathlib.Path(config_file.name)

    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = (
        config_filepath.parent /
        f"{config_filepath.name}-run-{time_str}.log"
    )
    logger_id = logger.add(log_path)

    logger.info(f"Running palom v{_version}")
    logger.info(f"Reading {config_file.name}\n\n{config_file.read().strip()}\n")
    config_file.close()

    config_data = yamale.make_data(config_file.name)
    if validate_config(config_data) == 1:
        return 1
    
    config = config_data[0][0]

    LEVEL = 0
    if 'pyramid level' in config:
        LEVEL = config['pyramid level']

    if 'pixel size' not in config:
        logger.warning(f"Pixel size in the output file is not set, using 1 µm")
        pixel_size = 1
    else: pixel_size = config['pixel size']

    images = get_image_list(config)
    
    image_paths = [
        pathlib.Path(config['input dir']) / pathlib.Path(i['filename'])
        for i in images
    ]

    image_output_modes = [
        i['output mode']
        for i in images
    ]

    channel_names = [
        i['channel name'] if 'channel name' in i else f"Channel {idx+1}"
        for idx, i in enumerate(images)
    ]

    output_path, qc_path = validate_output_path(config['output full path'])

    run_result = run_palom(
        img_paths=image_paths,
        img_modes=image_output_modes,
        pixel_size=pixel_size,
        channel_names=channel_names,
        output_path=output_path,
        qc_path=qc_path,
        level=LEVEL
    )

    logger.info(f"Finishing {config_file.name}")    
    logger.remove(logger_id)
    # Can't `pathlib.Path.replace` across different disk drive
    shutil.copy(log_path, qc_path / f"{output_path.name}.log")
    log_path.unlink()

    return run_result


def run_palom(
    img_paths,
    img_modes,
    pixel_size,
    channel_names,
    output_path,
    qc_path,
    level
):
    ref_reader = reader.SvsReader(img_paths[0])

    aligners = []
    for idx, p in enumerate(img_paths[1:]):
        logger.info(f"Processing {p.name}")
        moving_reader = reader.SvsReader(p)

        aligner = align.ReaderAligner(ref_reader, moving_reader, pyramid_level=level)
        aligner.coarse_register_affine()
        plt.suptitle(f"L: {aligner.ref_reader.path.name}\nR: {p.name}")
        plt.savefig(qc_path / f"{idx+1:02d}-{p.name}.png")
        aligner.compute_shifts()
        aligner.constrain_shifts()

        aligners.append(aligner)
    
    m1 = align.ReaderAligner(
        ref_reader, ref_reader, pyramid_level=level
    ).get_ref_mosaic(mode=img_modes[0])
    mosaics = [m1]
    mosaics += [
        a.get_aligned_mosaic(mode=m)
        for a, m in zip(aligners, img_modes[1:])
    ]

    write_pyramid.write_pyramid(
        mosaics, output_path,
        pixel_size=pixel_size,
        channel_names=channel_names
    )
    return 0


def validate_output_path(out_path, overwrite=True):
    # write access
    out_img_path = pathlib.Path(out_path)
    if out_img_path.exists():
        logger.warning(f"{out_img_path} already exists and will be overwritten")
    out_qc_path = out_img_path.parent / 'qc'
    out_qc_path.mkdir(exist_ok=True, parents=True)
    # folder is not empty
    if any(out_qc_path.iterdir()):
        logger.warning(f"{out_qc_path} already exists and will be overwritten")
    return out_img_path, out_qc_path


def validate_config(config_data):
    try:
        yamale.validate(schema.svs_config_schema, config_data)
        logger.info('Config YAML validation success')
        return 0
    except yamale.YamaleError as e:
        logger.info('Config YAML validation failed')
        for result in e.results:
            logger.error(f"Error validating data '{result.data}' with '{result.schema}'")
            for error in result.errors:
                logger.error(f'\t{error}')
        return 1


def get_image_list(config):
    out_list = [config['reference image']]
    if 'moving images' not in config:
        return out_list
    else:
        if type(config['moving images']) == list:
            return out_list + config['moving images']
        else:
            return out_list + [config['moving images']]


if __name__ == '__main__':
    sys.exit(main())