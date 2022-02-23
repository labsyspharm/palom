import numpy as np
import skimage.transform

import cv2
import itertools


def block_affine_dask(
    affine_matrix,
    src_array=None,
    fill_empty=0,
    multichannel=False,
    block_info=None,
):
    if not multichannel:
        mx = affine_matrix
        Y, X = block_info[None]['array-location']
    else:
        mx = affine_matrix[0]
        C, Y, X = block_info[None]['array-location']
    transformation = skimage.transform.AffineTransform(matrix=mx)
    shape = block_info[None]['chunk-shape']
    return block_affine(
        (Y[0], X[0]), shape,
        transformation, src_array,
        fill_empty=fill_empty, multichannel=multichannel
    )


def block_affine(
    position, block_shape, 
    transformation, src_img,
    fill_empty=0, multichannel=False
):
    assert np.min(block_shape) >= 0, (
        f'block_shape {block_shape} is invalid' 
    )
    if multichannel:
        assert np.min(block_shape) == block_shape[0], (
            'multichannel block must has shape of (C, Y, X)'
        )
        assert np.min(src_img.shape) == src_img.shape[0], (
            'multichannel image must be shaped as (C, Y, X)'
        )
        channel, height, width = block_shape
        block_shape = (height, width, channel)
    else:
        assert len(src_img.shape) == 2, (
            'src_img has more than 2 dimensions, use '
            'multichannel=True for color image'
        )
        height, width = block_shape
    y0_dst, x0_dst = position
    y1_dst, x1_dst = ((height, width) + np.array(position)).astype(int)

    inversed_corners = transformation.inverse(
        list(itertools.product(
            [x0_dst, x1_dst], [y0_dst, y1_dst]
        ))
    )

    x1_src, y1_src = np.ceil(
        np.clip(inversed_corners.max(axis=0), 0, None)
    ).astype(int)
    x0_src, y0_src = np.floor(
        np.clip(inversed_corners.min(axis=0), 0, None)
    ).astype(int)
    
    if multichannel:
        src_img_block = src_img[:, y0_src:y1_src, x0_src:x1_src]
        src_img_block = np.moveaxis(src_img_block, 0, 2)
    else:
        src_img_block = src_img[y0_src:y1_src, x0_src:x1_src]
    src_dtype = src_img.dtype
    if 0 in src_img_block.shape:
        return np.ones(block_shape, dtype=src_dtype) * fill_empty
    translation_offset = transformation([[x0_src, y0_src]]) - (x0_dst, y0_dst)

    block_tform = skimage.transform.AffineTransform(
        translation=translation_offset,
        scale=transformation.scale,
        rotation=transformation.rotation,
        shear=transformation.shear
    )
    warped_src_img_block = cv2.warpAffine(
        src_img_block, block_tform.params[:2, :],
        (width, height), flags=cv2.INTER_AREA
    )
    if multichannel:
        # shape multichannel image as (C, Y, X)
        warped_src_img_block = np.moveaxis(warped_src_img_block, 2, 0)

    return warped_src_img_block
