# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
visualizer class for filter scripts 
"""

import numpy as np
import pyvista as pv
from matplotlib import cm
from pyvistaqt import BackgroundPlotter
import trimesh
import torch
import copy
from os import path as osp
from modules.misc import DIRS
from viz.helpers import draw_poses
import queue
from PIL import Image
import tkinter as tk

pv.set_plot_theme("document")


class Viz:
    def __init__(
        self, off_screen: bool = False, zoom: float = 1.0, window_size: int = 0.5
    ):

        pv.global_theme.multi_rendering_splitting_position = 0.7
        """
            subplot(0, 0) main viz
            subplot(0, 1): tactile image viz
            subplot(1, 1): tactile codebook viz 
        """
        shape, row_weights, col_weights = (1, 2), [1.0], [0.6, 0.4]
        groups = [(np.s_[:], 0), (0, 1)]

        w, h = tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight()

        if off_screen:
            window_size = 1.0
        self.plotter = BackgroundPlotter(
            title="MidasTouch",
            lighting="three lights",
            window_size=(int(w * window_size), int(h * window_size)),
            off_screen=off_screen,
            shape=shape,
            row_weights=row_weights,
            col_weights=col_weights,
            groups=groups,
            border_color="white",
            toolbar=False,
            menu_bar=False,
            auto_update=True,
        )
        self.zoom = zoom

        self.viz_queue = queue.Queue(1)
        self.plotter.add_callback(self.update_viz, interval=500)
        self.pause = False
        self.font_size = int(30 * window_size)
        self.off_screen = off_screen

    def toggle_vis(self, flag):
        self.mesh_actor.SetVisibility(flag)

    def pause_vis(self, flag):
        self.pause = flag

    def set_camera(self, position="yz", azimuth=45, elevation=20, zoom=None):
        (
            self.plotter.camera_position,
            self.plotter.camera.azimuth,
            self.plotter.camera.elevation,
        ) = (position, azimuth, elevation)
        if zoom is None:
            self.plotter.camera.Zoom(self.zoom)
        else:
            self.plotter.camera.Zoom(zoom)
        self.plotter.camera_set = True

    def mirror_view(self):
        self.plotter.subplot(0, 0)
        cam = self.plotter.camera.copy()
        # self.plotter.subplot(1, 1)
        # self.plotter.camera = cam
        # self.plotter.camera.Zoom(0.8)

    def reset_vis(self, flag):
        self.plotter.subplot(0, 0)
        self.set_camera()
        self.reset_widget.value = not flag

    def init_variables(
        self,
        mesh_path: str,
        gt_pose: torch.Tensor = None,
        frame_rate: int = 30,
    ):
        self.mesh_pv = pv.read(mesh_path)  # pyvista object
        self.mesh_pv_deci = pv.read(
            mesh_path.replace("nontextured", "nontextured_decimated")
        )  # decimated pyvista object
        self.frame_rate = frame_rate
        self.moving_sensor = pv.read(
            osp.join(DIRS["obj_models"], "digit", "digit.STL")
        )  # plotted gt sensor
        self.init_sensor = copy.deepcopy(self.moving_sensor)  # sensor @ origin

        # Filter window
        self.plotter.subplot(0, 0)
        dargs = dict(
            color="grey",
            ambient=0.6,
            opacity=0.5,
            smooth_shading=True,
            specular=1.0,
            show_scalar_bar=False,
            render=False,
        )
        self.mesh_actor = self.plotter.add_mesh(self.mesh_pv, **dargs)

        if not self.off_screen:
            pos, offset = self.plotter.window_size[1] - 40, 10
            widget_size = 25
            self.plotter.add_checkbox_button_widget(
                self.toggle_vis,
                value=True,
                color_off="white",
                color_on="black",
                position=(10, pos),
                size=widget_size,
            )
            self.plotter.add_text(
                "Toggle object",
                position=(15 + widget_size, pos),
                color="black",
                font="times",
                font_size=self.font_size,
            )
            self.reset_widget = self.plotter.add_checkbox_button_widget(
                self.reset_vis,
                value=True,
                color_off="white",
                color_on="white",
                background_color="gray",
                position=(10, pos - (widget_size + offset)),
                size=widget_size,
            )
            self.plotter.add_text(
                "Reset camera",
                position=(15 + widget_size, pos - (widget_size + offset)),
                color="black",
                font="times",
                font_size=self.font_size,
            )
            self.plotter.add_checkbox_button_widget(
                self.pause_vis,
                value=False,
                color_off="white",
                color_on="black",
                position=(10, pos - 2 * (widget_size + offset)),
                size=widget_size,
            )
            self.plotter.add_text(
                "Pause",
                position=(15 + widget_size, pos - 2 * (widget_size + offset)),
                color="black",
                font="times",
                font_size=self.font_size,
            )
        self.set_camera()
        

        dargs = dict(
            color="tan",
            ambient=0.0,
            opacity=0.7,
            smooth_shading=True,
            show_edges=False,
            specular=1.0,
            show_scalar_bar=False,
            render=False,
        )
        self.plotter.add_mesh(self.moving_sensor, **dargs)

        if gt_pose is not None:
            gt_pose = np.atleast_3d(gt_pose)
            traj_plot = gt_pose[:, 0:3, 3].copy()
            faces_as_array = self.mesh_pv_deci.faces.reshape(
                (self.mesh_pv_deci.n_faces, 4)
            )[:, 1:]
            tmesh = trimesh.Trimesh(self.mesh_pv_deci.points, faces_as_array)
            (traj_plot, _, _) = trimesh.proximity.closest_point(tmesh, traj_plot)
            self.traj = pv.Spline(traj_plot, traj_plot.shape[0])
            self.traj["traversed"] = 0.5 * np.ones(self.traj.n_points)
            dargs = dict(
                cmap=cm.get_cmap("Greys"),
                scalars="traversed",
                interpolate_before_map=False,
                opacity=1.0,
                show_scalar_bar=False,
                silhouette=False,
                line_width=3,
                clim=[0.0, 1.0],
            )
            self.plotter.add_mesh(self.traj, **dargs)
        

        # Tactile window
        self.plotter.subplot(0, 1)
        self.plotter.camera.Zoom(1)
        self.plotter.add_text(
            "Tactile image and heightmap",
            position="bottom",
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="Tactile text",
        )

        self.viz_count = 0
        self.n_clusters = 0
        self.images = {"im": [], "path": []}
        self.image_plane, self.heightmap_plane = None, None

    def update_viz(
        self,
    ):
        if self.viz_queue.qsize():
            (
                gt_pose,
                image,
                heightmap,
                mask,
                frame,
                image_savepath,
            ) = self.viz_queue.get()
            # self.viz_filter(gt_pose, particles, cluster_poses, cluster_stds, frame)
            # self.viz_heatmap(heatmap_points, heatmap_weights)
            self.viz_tactile_image(image, heightmap, mask)
            self.viz_sensor(gt_pose)
            self.mirror_view()
            self.plotter.add_text(
                f"\nFrame {frame}   ",
                position="upper_right",
                color="black",
                shadow=True,
                font="times",
                font_size=self.font_size,
                name="frame text",
                render=True,
            )
            if image_savepath:
                self.images["im"].append(self.plotter.screenshot())
                self.images["path"].append(image_savepath)
            self.viz_queue.task_done()

    def update(
        self,
        gt_pose: np.ndarray,
        image: np.ndarray,
        heightmap: np.ndarray,
        mask: np.ndarray,
        frame: int,
        image_savepath: str = None,
    ) -> None:

        if self.viz_queue.full():
            self.viz_queue.get()
        self.viz_queue.put(
            (
                gt_pose,
                image,
                heightmap,
                mask,
                frame,
                image_savepath,
            ),
            block=False,
        )

    def viz_sensor(self, gt_pose: np.ndarray):
        transformed_gelsight_mesh = self.init_sensor.transform(gt_pose, inplace=False)
        self.moving_sensor.shallow_copy(transformed_gelsight_mesh)  
    
    def viz_tactile_image(
        self,
        image: np.ndarray,
        heightmap: torch.Tensor,
        mask: torch.Tensor,
        s: float = 1.8e-3,
    ) -> None:
        if self.image_plane is None:
            self.image_plane = pv.Plane(
                i_size=image.shape[1] * s,
                j_size=image.shape[0] * s,
                i_resolution=image.shape[1] - 1,
                j_resolution=image.shape[0] - 1,
            )
            self.image_plane.points[:, -1] = 0.25
            self.heightmap_plane = copy.deepcopy(self.image_plane)

        # visualize gelsight image
        self.plotter.subplot(0, 1)
        heightmap, mask = heightmap, mask
        image_tex = pv.numpy_to_texture(image)

        heightmap_tex = pv.numpy_to_texture(-heightmap * mask.astype(np.float32))
        self.heightmap_plane.points[:, -1] = (
            np.flip(heightmap * mask.astype(np.float32), axis=0).ravel() * (0.5 * s)
            - 0.15
        )
        self.plotter.add_mesh(
            self.image_plane,
            texture=image_tex,
            smooth_shading=False,
            show_scalar_bar=False,
            name="image",
            render=False,
        )
        self.plotter.add_mesh(
            self.heightmap_plane,
            texture=heightmap_tex,
            cmap=cm.get_cmap("plasma"),
            show_scalar_bar=False,
            name="heightmap",
            render=False,
        )

    def close(self):
        if len(self.images):
            for (im, path) in zip(self.images["im"], self.images["path"]):
                im = Image.fromarray(im.astype("uint8"), "RGB")
                im.save(path)

        self.plotter.close()
