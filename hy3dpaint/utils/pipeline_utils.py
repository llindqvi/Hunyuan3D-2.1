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

import torch
import numpy as np


class ViewProcessor:
    def __init__(self, config, render):
        self.config = config
        self.render = render

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(elev, azim, use_abs_coor=use_abs_coor, return_type="pl")
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(elev, azim, return_type="pl")
            position_maps.append(position_map)

        return position_maps

    # A view earns its slot only if it adds this fraction of the bakeable surface.
    min_view_texel_gain = 0.005

    def _candidate_trust_maps(self, candidate_camera_elevs, candidate_camera_azims):
        """Texels each candidate view would bake, from geometry alone.

        back_project's cosine map depends only on the mesh and the camera, so the
        footprint is known before any view image exists. The probe is rendered at
        the same resolution the real bake uses -- a smaller one scatters fewer
        pixels into the texture and would under-report coverage.
        """
        probe = np.zeros((self.config.render_size, self.config.render_size), dtype=np.float32)
        trust_maps = []
        for elev, azim in zip(candidate_camera_elevs, candidate_camera_azims):
            _, cos_map, _ = self.render.back_project(probe, elev, azim)
            trust_maps.append(cos_map.squeeze(-1) > 0)
        return trust_maps

    def bake_view_selection(
        self, candidate_camera_elevs, candidate_camera_azims, candidate_view_weights, max_selected_view_num
    ):
        """Pick the views to paint from, ranked by the texels they actually bake.

        Ranking by newly-*rasterized triangle area* -- the previous criterion --
        counts a triangle as covered once any view catches a single pixel of it at
        any grazing angle. That proxy saturates right after the axis views, so the
        search stopped at six views while a tenth of the surface still reached no
        camera; those texels became inpaint blotches in concave regions (inner
        legs, armpits). Trusted-texel coverage is what the bake consumes, so rank
        on it directly and let max_selected_view_num be the real bound.

        The front view is always kept: it holds the highest bake weight and anchors
        the texture to the reference image, and coverage alone would discard it.
        """
        trust_maps = self._candidate_trust_maps(candidate_camera_elevs, candidate_camera_azims)
        bakeable = torch.stack(trust_maps).any(dim=0).sum().clamp(min=1)

        selected = [0]
        covered = trust_maps[0].clone()
        while len(selected) < max_selected_view_num:
            gains = [
                (int((trust_maps[idx] & ~covered).sum()), idx)
                for idx in range(len(trust_maps))
                if idx not in selected
            ]
            if not gains:
                break
            gain, best_idx = max(gains)
            if gain / bakeable < self.min_view_texel_gain:
                break
            selected.append(best_idx)
            covered |= trust_maps[best_idx]

        return (
            [candidate_camera_elevs[idx] for idx in selected],
            [candidate_camera_azims[idx] for idx in selected],
            [candidate_view_weights[idx] for idx in selected],
        )

    def bake_from_multiview(self, views, camera_elevs, camera_azims, view_weights):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []

        for view, camera_elev, camera_azim, weight in zip(views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim
            )
            project_cos_map = weight * (project_cos_map**self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)
            texture, ori_trust_map = self.render.fast_bake_texture(project_textures, project_weighted_cos_maps)
        return texture, ori_trust_map > 1e-8

    def texture_inpaint(self, texture, mask, defualt=None):
        if defualt is not None:
            mask = mask.astype(bool)
            inpaint_value = torch.tensor(defualt, dtype=texture.dtype, device=texture.device)
            texture[~mask] = inpaint_value
        else:
            texture_np = self.render.uv_inpaint(texture, mask)
            texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture
