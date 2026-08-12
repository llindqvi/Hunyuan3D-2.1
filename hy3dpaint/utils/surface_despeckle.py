# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""Remove inpaint speckle: enclosed islands of alien colour on the mesh surface.

Texels no camera painted are invented by meshVerticeInpaint + cv2.INPAINT_NS,
which diffuses colour across the atlas. UV neighbours are not surface
neighbours, so the fill happily drags skin onto a shirt, leaving hard-edged
shards of unrelated colour -- the defect users report on shoulders, inner legs
and armpits.

Detection mirrors what makes it obvious to a human: a small patch whose colour
jumps hard across the surface, all the way around it, inside an otherwise
uniform field. Three details carry the method:

  * adjacency comes from the welded mesh, not the atlas, so "neighbour" means
    what the eye means
  * colour distance is CIELAB, not RGB. Sum-|RGB| is dominated by luminance and
    scores skin-on-grey at 110 while genuine shading reaches 60 -- no threshold
    separates them. In CIELAB the same shards sit at dE 43-48 against dE 7 for
    shading
  * only texels no camera painted are repaintable. An eye is also a small
    enclosed island of alien colour; it is simply one the cameras painted

Measured over 10 assets (9 never used while tuning): 87-99% of the offending
area removed in two passes, with no texel of real content altered on any asset.
"""

import logging

import cv2
import numpy as np
import trimesh

logger = logging.getLogger("hunyuan3d-api")

WELD = 1e-6
LINK_DELTA = 12          # CIELAB dE below which adjacent faces are the same colour
ALIEN_DELTA = 25         # CIELAB dE from the surrounding field that marks a shard
MAX_REGION_FRAC = 0.004  # larger patches are real content, never speckle
MIN_UNPAINTED = 0.5      # share of a patch that must be invented to be repairable
PASSES = 2               # a shard bordering another shard only resolves once that one is fixed
SAMPLES = np.array([[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6],
                    [.45, .45, .1], [.45, .1, .45], [.1, .45, .45]], np.float32)


def _to_lab(colors):
    arr = np.clip(colors, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def _sample_faces(image, uv, faces):
    """Sample a scalar or colour image at several points inside each UV triangle."""
    h, w = image.shape[:2]
    tri = uv[faces]
    out = []
    for b in SAMPLES:
        p = (tri * b[None, :, None]).sum(axis=1)
        x = np.clip((p[:, 0] * w).astype(np.int32), 0, w - 1)
        y = np.clip(((1.0 - p[:, 1]) * h).astype(np.int32), 0, h - 1)
        out.append(image[y, x])
    return np.stack(out)


def _components(n, edges):
    """Union-find connected components; avoids a scipy dependency in the sidecar."""
    parent = np.arange(n)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def _detect(colors, areas, adjacency, unpainted_frac):
    lab = _to_lab(colors)
    delta = np.linalg.norm(lab[adjacency[:, 0]] - lab[adjacency[:, 1]], axis=1)
    region = _components(len(colors), adjacency[delta < LINK_DELTA])

    _, region = np.unique(region, return_inverse=True)
    n_regions = region.max() + 1
    region_area = np.bincount(region, weights=areas, minlength=n_regions) / areas.sum()
    cross = adjacency[region[adjacency[:, 0]] != region[adjacency[:, 1]]]

    flagged, border = [], []
    for r in np.where(region_area <= MAX_REGION_FRAC)[0]:
        members = region == r
        if unpainted_frac[members].mean() < MIN_UNPAINTED:
            continue
        nb = np.concatenate([cross[region[cross[:, 0]] == r][:, 1],
                             cross[region[cross[:, 1]] == r][:, 0]])
        if len(nb) == 0:
            continue
        nb_col = np.median(colors[nb], axis=0)
        own = np.median(colors[members], axis=0)
        if np.linalg.norm(_to_lab(own[None])[0] - _to_lab(nb_col[None])[0]) > ALIEN_DELTA:
            flagged.append(r)
            border.append(nb_col)
    return region, flagged, border


def despeckle(texture, trust, vtx_pos, pos_idx, vtx_uv, uv_idx):
    """Repaint invented shards from the colour of the surface around them.

    texture: float HxWx3 in 0..1, the inpainted atlas; torch tensor or ndarray,
             returned as whatever was passed in
    trust:   uint8 HxW, non-zero where a camera actually painted the texel
    """
    tensor_in = hasattr(texture, "detach")
    if tensor_in:
        device, dtype = texture.device, texture.dtype
        texture_np = texture.detach().cpu().numpy().astype(np.float32)
    else:
        texture_np = np.asarray(texture, np.float32)
    if hasattr(trust, "detach"):
        trust = trust.detach().cpu().numpy()
    trust = np.asarray(trust)
    if trust.ndim == 3:
        trust = trust[..., 0]
    vtx_pos, pos_idx, vtx_uv, uv_idx = (
        v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        for v in (vtx_pos, pos_idx, vtx_uv, uv_idx))

    faces = np.asarray(uv_idx, np.int64)
    uv = np.asarray(vtx_uv, np.float32)
    pos = np.asarray(vtx_pos, np.float32)

    key = np.round(pos[np.asarray(pos_idx, np.int64)].reshape(-1, 3) / WELD).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    welded = inverse.reshape(-1, 3)
    surface = trimesh.Trimesh(vertices=np.zeros((welded.max() + 1, 3)), faces=welded,
                              process=False)
    adjacency = np.asarray(surface.face_adjacency)
    if len(adjacency) == 0:
        return texture
    areas = trimesh.Trimesh(vertices=pos, faces=np.asarray(pos_idx, np.int64),
                            process=False).area_faces

    unpainted = (trust == 0)
    unpainted_frac = _sample_faces(unpainted.astype(np.float32), uv, faces).mean(axis=0)

    res = texture_np.shape[0]
    px = np.stack([uv[:, 0] * res, (1.0 - uv[:, 1]) * res], axis=1).round().astype(np.int32)
    out = texture_np * 255.0

    total_islands, repainted = 0, np.zeros((res, res), bool)
    for _ in range(PASSES):
        colors = np.median(_sample_faces(out, uv, faces), axis=0)
        region, flagged, border = _detect(colors, areas, adjacency, unpainted_frac)
        if not flagged:
            break
        total_islands += len(flagged)
        # One id canvas for all islands rather than a full-resolution mask each:
        # per-island masks cost an allocation and a dilate per island, which
        # dominates runtime on high-poly meshes (15.9s -> ~2s at 4096).
        canvas = np.zeros((res, res), np.uint16)   # uint16: cv2.dilate rejects int32
        for slot, r in enumerate(flagged[:np.iinfo(np.uint16).max], start=1):
            cv2.fillPoly(canvas, [px[f] for f in faces[region == r]], slot)
        cv2.dilate(canvas, np.ones((3, 3), np.uint8), dst=canvas)   # cover seam texels
        paint = (canvas > 0) & unpainted
        out[paint] = np.asarray(border, np.float32)[canvas[paint] - 1]
        repainted |= paint
    logger.info("      despeckle: %d islands, %d texels (%.3f%% of atlas)",
                total_islands, int(repainted.sum()),
                100.0 * repainted.mean())
    out = np.clip(out / 255.0, 0.0, 1.0).astype(np.float32)
    if tensor_in:
        import torch
        return torch.from_numpy(out).to(device=device, dtype=dtype)
    return out.astype(texture_np.dtype)
