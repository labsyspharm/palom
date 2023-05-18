import sys
import argparse
import pathlib
import yamale
from loguru import logger
import datetime
import shutil

from . import schema
from .. import reader, align, pyramid, color
from .. import __version__ as _version

import matplotlib.pyplot as plt
import numpy as np


logger.remove()  # All configured handlers are removed
logger.add(
    sys.stderr,
    format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>'
)


@logger.catch
def main(argv=sys.argv):

    parser = argparse.ArgumentParser(
        description=(
            'Align multiple SVS images of the same biospecimen and write a merged'
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

    if 'pixel size' in config:
        pixel_size = config['pixel size']
        logger.info(f"Using pixel size defined in configuration YAML file: {pixel_size} µm/pixel")
    else: pixel_size = None

    DOWNSCALE_FACTOR = 4
    if 'pyramid downscale factor' in config:
        DOWNSCALE_FACTOR = config['pyramid downscale factor']

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

    channel_names = []
    for idx, i in enumerate(images):
        if 'channel names' in i:
            names = i['channel names']
        elif 'channel name' in i:
            names = [i['channel name']]
        else:
            names = [f"File {idx+1}"]
        channel_names.append(names)

    output_path, qc_path = validate_output_path(config['output full path'])

    run_result = run_palom(
        img_paths=image_paths,
        img_modes=image_output_modes,
        pixel_size=pixel_size,
        channel_names=channel_names,
        output_path=output_path,
        qc_path=qc_path,
        level=LEVEL,
        downscale_factor=DOWNSCALE_FACTOR
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
    level,
    downscale_factor
):
    ref_reader = reader.SvsReader(img_paths[0])
    ref_color_proc = color.PyramidHaxProcessor(ref_reader.pyramid)
    ref_thumbnail_level = ref_reader.get_thumbnail_level_of_size(2500)

    block_affines = []
    for idx, p in enumerate(img_paths[1:]):
        logger.info(f"Processing {p.name}")
        if p == img_paths[0]:
            block_affines.append(np.eye(3))
            continue
        moving_reader = reader.SvsReader(p)
        moving_color_proc = color.PyramidHaxProcessor(moving_reader.pyramid)
        moving_thumbnail_level = moving_reader.get_thumbnail_level_of_size(2500)

        aligner = align.Aligner(
            ref_color_proc.get_processed_color(level, 'grayscale'),
            moving_color_proc.get_processed_color(level, 'grayscale'),
            ref_color_proc.get_processed_color(ref_thumbnail_level, 'grayscale').compute(),
            moving_color_proc.get_processed_color(moving_thumbnail_level, 'grayscale').compute(),
            ref_reader.level_downsamples[ref_thumbnail_level] / ref_reader.level_downsamples[level],
            moving_reader.level_downsamples[moving_thumbnail_level] / moving_reader.level_downsamples[level]
        )

        aligner.coarse_register_affine()
       
        # FIXME move the saving figure logic
        plt.suptitle(f"L: {ref_reader.path.name}\nR: {p.name}")
        fig_w = max(plt.gca().get_xlim())
        fig_h = max(plt.gca().get_ylim()) + 100
        factor = 1600 / max(fig_w, fig_h)
        plt.gcf().set_size_inches(fig_w*factor/72, fig_h*factor/72)
        plt.tight_layout()
        plt.savefig(qc_path / f"{idx+1:02d}-{p.name}.png", dpi=72)
        plt.close()
       
        aligner.compute_shifts()
        aligner.constrain_shifts()

        block_affines.append(aligner.block_affine_matrices_da)
   
    mosaics = []
    m_ref = ref_color_proc.get_processed_color(level=level, mode=img_modes[0])
    mosaics.append(m_ref)
    for p, m, mx in zip(img_paths[1:], img_modes[1:], block_affines):
        moving_color_proc = color.PyramidHaxProcessor(
            reader.SvsReader(p).pyramid
        )
        m_moving = align.block_affine_transformed_moving_img(
            ref_color_proc.get_processed_color(level),
            moving_color_proc.get_processed_color(level, mode=m),
            mx
        )
        mosaics.append(m_moving)

    if pixel_size is None:
        pixel_size = ref_reader.pixel_size

    pyramid.write_pyramid(
        mosaics,
        output_path,
        pixel_size=pixel_size,
        channel_names=channel_names,
        downscale_factor=downscale_factor
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
    plt.switch_backend('agg')
    sys.exit(main())