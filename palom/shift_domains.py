"""Partition a block-shift field into domains of mutually-agreeing shifts.

The problem this solves: on a slide carrying several tissue pieces, each piece
sits at its own rigid offset -- 50-150 px measured across the reference set --
because ashlar places every non-primary connected component by centroid-matching
a model extrapolated from raw stage coordinates, translation only. A stitching
error does the same thing *inside* one piece, at tile boundaries.

The two images are the *same physical section*, so the pieces cannot have moved
relative to each other: every per-piece offset is a stitching artifact, and is
therefore translation-only by construction rather than by measurement.

Neither is representable by one affine plus a C0 displacement field, and
`align.constrain_block_shifts` actively regresses the second domain away: it
fits one plane over the whole grid and thresholds residuals with
`threshold_triangle`, which assumes a unimodal histogram. Given two offsets the
threshold lands between the modes and the minority domain is overwritten by the
plane's prediction.

The partition here is ashlar's, from `rc/align_to_stitched.StitchedLayerAligner`:
keep a grid-adjacency edge only where two neighbouring blocks' *measured shifts
agree*, and take connected components. It separates tissue pieces and stitching
seams with one mechanism, and it works on measured agreement rather than
morphology -- so it finds a seam that runs through the middle of one connected
piece, which no segmentation can.

Agreement is deliberately the criterion rather than a per-block correlation
score: the correlation error is an absolute quantity, contaminated by texture,
stain and tissue type, so a threshold on it does not transfer between slides.
Agreement is relative and local, and its threshold is a physical displacement.
"""

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

# Two independent measurements that agree are the minimum corroboration; a lone
# block is not. Straight from ashlar's `_min_component_size`.
MIN_DOMAIN_BLOCKS = 2

LOOSE = -1

# Default agreement tolerance, in the same pixels as the shifts. See
# `align_multi_obj.DEFAULT_DOMAIN_TOL` for how it was chosen.
DEFAULT_TOL = 20.0

# Below this share of a region resolved into domains, the partition is islands
# in noise rather than a partition, and its offsets should not be read as
# findings. Not a reason to stop: ImageLSP23390 covers 37% because tissue was
# genuinely lost between the two scans, and the run is still wanted -- it is a
# reason to flag the slide for review. Provisional, set from four slides
# (37% / 93% / 98% / 100%).
LOW_COVERAGE = 0.75


def _agreement_edges(field, valid, tol):
    """Row/column neighbour pairs whose shifts agree within `tol`.

    Returns flat index pairs into the ``h * w`` grid.
    """
    h, w = valid.shape
    idx = np.arange(h * w).reshape(h, w)
    pairs = []
    for a, b, va, vb in (
        (field[:-1, :], field[1:, :], valid[:-1, :], valid[1:, :]),   # vertical
        (field[:, :-1], field[:, 1:], valid[:, :-1], valid[:, 1:]),   # horizontal
    ):
        keep = va & vb & (np.linalg.norm(a - b, axis=-1) <= tol)
        if a.shape[0] < h:
            ia, ib = idx[:-1, :][keep], idx[1:, :][keep]
        else:
            ia, ib = idx[:, :-1][keep], idx[:, 1:][keep]
        pairs.append((ia, ib))
    return (
        np.concatenate([p[0] for p in pairs]),
        np.concatenate([p[1] for p in pairs]),
    )


def label_domains(shifts, grid_shape, valid=None, tol=DEFAULT_TOL,
                  min_size=MIN_DOMAIN_BLOCKS):
    """Label blocks by domain; `LOOSE` (-1) where no domain could be trusted.

    `shifts` is ``(n_blocks, 2)`` row-major over `grid_shape`, `valid` a
    per-block bool (default: every finite block). `tol` is the largest
    neighbour-to-neighbour shift difference still considered one domain, in the
    same pixels as `shifts`.

    Domains are renumbered largest-first, so domain 0 is always the biggest --
    the one loose blocks are most likely to be resolved against.
    """
    h, w = grid_shape
    field = np.asarray(shifts, dtype=float).reshape(h, w, 2)
    finite = np.isfinite(field).all(axis=-1)
    valid = finite if valid is None else (np.asarray(valid, bool).reshape(h, w) & finite)

    ia, ib = _agreement_edges(field, valid, tol)
    adj = scipy.sparse.coo_matrix(
        (np.ones(len(ia)), (ia, ib)), shape=(h * w, h * w)
    )
    _, raw = scipy.sparse.csgraph.connected_components(adj, directed=False)
    raw = np.where(valid.ravel(), raw, LOOSE)

    labels = np.full(h * w, LOOSE)
    keep = [c for c in np.unique(raw[raw != LOOSE])
            if (raw == c).sum() >= min_size]
    # largest first, so domain 0 is the dominant one
    keep.sort(key=lambda c: -(raw == c).sum())
    for new, old in enumerate(keep):
        labels[raw == old] = new
    return labels.reshape(h, w)


def domain_offsets(shifts, labels):
    """Median shift per domain, as ``{label: array([dy, dx])}``.

    The median rather than the mean: a domain's blocks are its own measurements
    plus whatever landed on its rim, and one rim block sitting across a seam
    should not drag the offset.
    """
    labels = np.asarray(labels)
    field = np.asarray(shifts, dtype=float).reshape(*labels.shape, 2)
    out = {}
    for lab in np.unique(labels[labels != LOOSE]):
        members = field[labels == lab]
        members = members[np.isfinite(members).all(axis=-1)]
        if members.size:
            out[int(lab)] = np.median(members, axis=0)
    return out


def summarize(shifts, labels, within=None):
    """One row per domain, plus how much of the region they cover.

    `spread` is the median distance of a domain's blocks from its own offset --
    how much genuine local deformation is left once the rigid part is removed.

    `within` bounds the accounting to the blocks that matter (an object's own
    blocks, say). Without it the loose count silently includes every background
    block in the grid, which is most of it -- an object of 787 blocks reported
    1375 loose before this argument existed.
    """
    labels = np.asarray(labels)
    field = np.asarray(shifts, dtype=float).reshape(*labels.shape, 2)
    within = (
        np.ones(labels.shape, bool)
        if within is None
        else np.asarray(within, bool).reshape(labels.shape)
    )
    offsets = domain_offsets(shifts, labels)
    rows = []
    for lab, off in sorted(offsets.items()):
        members = field[labels == lab]
        members = members[np.isfinite(members).all(axis=-1)]
        rows.append({
            "domain": lab,
            "blocks": int((labels == lab).sum()),
            "offset": off,
            "spread": float(np.median(np.linalg.norm(members - off, axis=-1))),
        })
    n_within = int(within.sum())
    n_loose = int((within & (labels == LOOSE)).sum())
    return {
        "domains": rows,
        "n_blocks": n_within,
        "n_loose": n_loose,
        # Share of the region a domain could be found for. A low value means
        # agreement had little to work with, and the domains found are islands
        # in noise rather than a partition -- read it before the offsets.
        "coverage": (n_within - n_loose) / n_within if n_within else 0.0,
    }


def max_separation(shifts, labels):
    """Largest offset between any two domains, 0.0 when there is one or none.

    The number that says whether the partition found anything worth acting on:
    a slide whose domains all sit within a pixel or two of each other did not
    need splitting.
    """
    offsets = list(domain_offsets(shifts, labels).values())
    if len(offsets) < 2:
        return 0.0
    return float(
        max(np.linalg.norm(a - b) for i, a in enumerate(offsets) for b in offsets[i + 1:])
    )
