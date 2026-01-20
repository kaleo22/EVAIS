#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template: Mehrfach ein eigenes USD-Objekt importieren und synthetische Kamerabilder + Instance-Segmentation-GT speichern.
Hinweis: Pfade/Parameter sind Platzhalter; muss nicht sofort laufen.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
import argparse

#from isaacsim.core.api.scenes import scene
import numpy as np
from PIL import Image

# # 1) Isaac Sim zuerst starten (wichtig: vor vielen IsaacLab/omni Imports)
# try:
#     from isaacsim import SimulationApp  # Isaac Sim API
# except Exception as e:
#     raise RuntimeError("Isaac Sim Python-Umgebung fehlt oder SimulationApp nicht importierbar.") from e

from isaaclab.app import AppLauncher

# create argparser
#parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
def parse_float_list(s: str) -> list[float]:
    s = s.strip().strip("[]").replace(" ", "")
    if not s:
        return []
    return [float(x) for x in s.split(",")]

p = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
p.add_argument("--num_instances", type=int, default=3, help="Wie oft das gleiche USD in die Stage gesetzt wird.")
p.add_argument("--output_dir", type=str, default="output_dataset", help="Output-Verzeichnis für das Dataset.")
p.add_argument("--wheel_spacing", type=parse_float_list, default="[5.0]", help="Abstände zwischen den Rädern in Metern (x).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(p)
# parse the arguments
args = p.parse_args()
NUM_INSTANCES = args.num_instances
spacings = parse_float_list(args.wheel_spacing) if isinstance(args.wheel_spacing, str) else args.wheel_spacing
args_cli = args  # p.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
simulation_app.set_setting("/rtx/enabled", True)
simulation_app.set_setting("/rtx/shadows/enabled", True)
simulation_app.set_setting("/rtx/reflections/enabled", True)

#simulation_app = SimulationApp({"headless": False})  # True für headless

# 2) Danach IsaacLab/IsaacSim-Module importieren
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.semantics import add_labels
#from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Optional: zum nachträglichen Umpositionieren
from isaacsim.core.prims import XFormPrim
import torch

# -----------------------------
# Konfiguration: dein USD + Output
# -----------------------------
USD_PATH = "/home/leonard/Downloads/rear_wheel.usd"
USD_PATH_2 = "/home/leonard/Downloads/front_wheel.usd"
OUTPUT_DIR = Path(__file__).resolve().parent / args.output_dir


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        mx = float(np.nanmax(img)) if img.size else 0.0
        if mx <= 1.0 + 1e-6:
            img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)
    return np.clip(img.astype(np.int64), 0, 255).astype(np.uint8)


# def _save_rgb_png(rgb: np.ndarray, path: Path) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     rgb = _to_uint8_rgb(rgb)
#     if rgb.ndim == 3 and rgb.shape[-1] >= 3:
#         rgb = rgb[..., :3]
#     Image.fromarray(rgb, mode="RGB").save(path)

def _save_rgb_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(rgb)
    while rgb.ndim > 3:
        rgb = rgb[0]  # drop batch/env dims (e.g. 1,H,W,4 -> H,W,4)

    rgb = _to_uint8_rgb(rgb)

    if rgb.ndim == 3 and rgb.shape[-1] >= 3:
        rgb = rgb[..., :3]

    Image.fromarray(rgb).save(path)


def _save_instance_seg(seg: np.ndarray, npy_path: Path, preview_png_path: Path | None = None) -> None:
    """
    Speichert Instance-Segmentation:
    - npy: rohe IDs (empfohlen als GT)
    - preview: simple farbige Visualisierung (optional)
    """
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, seg)

    if preview_png_path is None:
        return

    preview_png_path.parent.mkdir(parents=True, exist_ok=True)

    # Erwartung: (H,W) oder (H,W,C). Falls C>1, nimm ersten Kanal als ID.
    if seg.ndim == 3:
        seg_ids = seg[..., 0]
    else:
        seg_ids = seg

    seg_ids = seg_ids.astype(np.int64)
    # simple, deterministisches Coloring
    r = (seg_ids * 37) % 255
    g = (seg_ids * 17) % 255
    b = (seg_ids * 29) % 255
    viz = np.stack([r, g, b], axis=-1).astype(np.uint8)

    Image.fromarray(viz, mode="RGB").save(preview_png_path)

def _quat_from_axis_angle(axis_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_xyz = np.asarray(axis_xyz, dtype=np.float64)
    n = np.linalg.norm(axis_xyz)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz
    axis_xyz = axis_xyz / n
    s = np.sin(angle_rad * 0.5)
    return np.array([np.cos(angle_rad * 0.5), axis_xyz[0] * s, axis_xyz[1] * s, axis_xyz[2] * s], dtype=np.float64)

def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ],
        dtype=np.float64,
    )

# def _set_xform_world_pose(prim: XFormPrim, pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
#     pos = tuple(float(v) for v in pos_xyz)
#     quat = tuple(float(v) for v in quat_wxyz)
#     if hasattr(prim, "set_world_poses"):
#         prim.set_world_poses(positions=pos, orientations=quat)
#     elif hasattr(prim, "set_local_pose"):
#         prim.set_local_pose(translation=pos, orientation=quat)
#     else:
#         raise AttributeError("XFormPrim has no set_world_pose/set_local_pose")
    
def _set_xform_world_pose(prim: XFormPrim, pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
    pos = (float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
    quat = (float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))

    if hasattr(prim, "set_world_poses"):
        positions = torch.tensor([pos], dtype=torch.float32)
        orientations = torch.tensor([quat], dtype=torch.float32)
        prim.set_world_poses(positions=positions, orientations=orientations)
        return

    if hasattr(prim, "set_local_pose"):
        prim.set_local_pose(translation=pos, orientation=quat)
        return

    raise AttributeError("XFormPrim has no set_world_poses/set_local_pose")

@dataclass
class PseudoVehicle:
    wheel_paths: list[str]
    v_mps: float = 1.0
    yaw_rate_rps: float = 0.0
    wheel_radius_m: float = 0.3
    spin_axis: str = "y"  # "x"|"y"|"z"
    base_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    yaw: float = 0.0
    wheel_angle: float = 0.0
    mount_quat_wxyz: np.ndarray = field(
    default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity
    )
    wheel_offsets: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.5,  0.4, 0.0],
                [0.5, -0.4, 0.0],
                [-0.5, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    _wheels: list[XFormPrim] = field(default_factory=list, init=False)

    def bind(self) -> None:
        self._wheels = [XFormPrim(p) for p in self.wheel_paths]

    def step(self, dt: float) -> None:
        dt = float(dt)
        self.yaw += self.yaw_rate_rps * dt

        c, s = np.cos(self.yaw), np.sin(self.yaw)
        Rz = np.array([[c, -s, 0.0],
                       [s,  c, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)

        forward = np.array([c, s, 0.0], dtype=np.float64)
        self.base_pos = self.base_pos + forward * (self.v_mps * dt)

        self.wheel_angle += (self.v_mps / max(self.wheel_radius_m, 1e-6)) * dt

        q_yaw = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), self.yaw)

        axis_local = {"x": np.array([1.0, 0.0, 0.0]),
                      "y": np.array([0.0, 1.0, 0.0]),
                      "z": np.array([0.0, 0.0, 1.0])}.get(self.spin_axis, np.array([0.0, 1.0, 0.0]))
        q_spin = _quat_from_axis_angle(axis_local, self.wheel_angle)

        q_total = _quat_mul_wxyz(_quat_mul_wxyz(q_yaw, self.mount_quat_wxyz), q_spin)

        for i, wheel in enumerate(self._wheels):
            off = self.wheel_offsets[i] if i < len(self.wheel_offsets) else np.array([0.0, 0.0, 0.0], dtype=np.float64)
            wheel_pos = self.base_pos + (Rz @ off)
            _set_xform_world_pose(wheel, wheel_pos, q_total)

def move_vehicles(vehicle: PseudoVehicle, dt: float) -> None:
    "Räder mit geschwindigkeit pro Frame bewegen"
    vehicle.step(dt)

# def create_wheels(spacings: list[float]) -> tuple[list[str], tuple[tuple[float, float, float], ...]]:
#         prim_paths = [""] * len(spacings)
#         spacing_tuple = tuple((float(s), 0.0, 0.0) for s in spacings)
#         for i, spacing in enumerate(spacings):
#             prim_path = f"/World/Objects/Wheel_{i}"
#             prim_paths[i] = prim_path
#             wheel_cfg = AssetBaseCfg(
#                 prim_path=prim_path,
#                 spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
#             )
#             scene.add_asset(f"wheel_{i}", wheel_cfg)
#             wheel_prim = XFormPrim(prim_path)
#             wheel_prim.set_local_pose(
#                 translation=(spacing, 0.0, 0.28),
#                 orientation=(1.0, 0.0, 0.0, 0.0),  # wxyz
#             )
#         return prim_paths, spacing_tuple

# def create_wheels(scene: InteractiveScene, spacings: list[float]) -> tuple[list[str], np.ndarray]:
#     prim_paths: list[str] = []
#     wheel_offsets: list[list[float]] = []

#     for i, x in enumerate(spacings):
#         prim_path = f"/World/Objects/Wheel_{i}"
#         prim_paths.append(prim_path)
#         wheel_offsets.append([float(x), 0.0, 0.0])

#         wheel_cfg = AssetBaseCfg(
#             prim_path=prim_path,
#             spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH,
#                                        semantic_tags=[("Class", "Wheel")])
#         )
#         #scene.add_asset(f"wheel_{i}", wheel_cfg)

#    return prim_paths, np.array(wheel_offsets, dtype=np.float64)

def create_wheels(scene: InteractiveScene, spacings: list[float]) -> tuple[list[str], np.ndarray]:
    prim_paths: list[str] = []
    wheel_offsets: list[list[float]] = []
    prim_path_front = f"/World/Objects/Wheel_Front"
    prim_paths.append(prim_path_front)
    wheel_offsets.append([0.0, 0.0, 0.0])
    front_wheel_cfg = AssetBaseCfg(
        prim_path=prim_path_front,
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH_2,
                                   scale=(1.0, 1.0, 1.0),
                                      semantic_tags=[("Class", "Wheel")])
    )
    prim_front = add_reference_to_stage(usd_path=USD_PATH_2, prim_path=prim_path_front, prim_type="Xform")
    add_labels(prim_front, ["Wheel"], instance_name="class", overwrite=True)

    for i, x in enumerate(spacings):
        prim_path = f"/World/Objects/Wheel_{i}"
        prim_paths.append(prim_path)
        wheel_offsets.append([float(x), 0.0, 0.0])

        wheel_cfg = AssetBaseCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH,
                             scale=(1.0, 1.0, 1.0),
                                semantic_tags=[("Class", "Wheel")])
        )

        prim = add_reference_to_stage(usd_path=USD_PATH, prim_path=prim_path, prim_type="Xform")
        #prim = sim_utils.create_prim(prim_path, prim_type="SkelRoot", usd_path=USD_PATH)
        add_labels(prim, ["Wheel"], instance_name="class", overwrite=True)

    return prim_paths, np.array(wheel_offsets, dtype=np.float64)

# def create_wheels(scene: InteractiveScene, spacings: list[float]) -> tuple[list[str], np.ndarray]:
#     prim_paths: list[str] = []
#     wheel_offsets: list[list[float]] = []

#     from pxr import UsdGeom

#     for i, x in enumerate(spacings):
#         prim_path = f"/World/Objects/Wheel_{i}"
#         prim_paths.append(prim_path)
#         wheel_offsets.append([float(x), 0.0, 0.0])

#         #prim = add_reference_to_stage(usd_path=USD_PATH, prim_path=prim_path, prim_type="Xform")
#         import omni.usd
#         from pxr import UsdGeom

#         stage = omni.usd.get_context().get_stage()

#         xform = UsdGeom.Xform.Define(stage, prim_path)
#         xform.GetPrim().GetReferences().AddReference(USD_PATH, "/World/truck_wheels_front_v2")
#         prim = xform.GetPrim()
#         prim.Load()

#         if prim:
#             prim.Load()
#             try:
#                 UsdGeom.Imageable(prim).MakeVisible()
#             except Exception:
#                 pass

#             add_labels(prim, ["Wheel"], instance_name="class", overwrite=True)

#     return prim_paths, np.array(wheel_offsets, dtype=np.float64)

@configclass
class DatasetSceneCfg(InteractiveSceneCfg):
    # Ground + Light (Beispiel)
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # dome_light = AssetBaseCfg(
    #     prim_path="/World/Light",
    #     spawn=sim_utils.DomeLightCfg(
    #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/HDRI/Indoor/Studio.hdr",
    #         intensity=1500.0,
    #     ),
    # )
    dome_light = AssetBaseCfg(
    prim_path="/World/Light",
    spawn=sim_utils.DomeLightCfg(
        intensity=1500.0,
    ),
    )

    # Kamera: liefert RGB + Instance Segmentation
    camera = TiledCameraCfg(
        prim_path="/World/Sensors/Camera",
        update_period=0.0,
        data_types=["rgb", "instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 50.0),
        ),
        width=1280,
        height=720,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(5.0, -4.2, 1),
            rot=(0.704416, 0.704416, 0.06163, 0.06163),  # quat (w,x,y,z) in usd-konvention; ggf. anpassen
            convention="usd",
        ),
    )

    # # Mehrfaches Spawnen deines USD: gleiche Datei, unterschiedliche prim_paths
    # # (Einfach duplizieren/erweitern, wenn du mehr willst.)
    # usd_obj_0 = AssetBaseCfg(
    #     prim_path="/World/Objects/MyUsd_0",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=USD_PATH,
    #         # scale=(1.0, 1.0, 1.0),  # optional
    #     ),
    # )
    # usd_obj_1 = AssetBaseCfg(
    #     prim_path="/World/Objects/MyUsd_1",
    #     spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
    # )
    # usd_obj_2 = AssetBaseCfg(
    #     prim_path="/World/Objects/MyUsd_2",
    #     spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
    # )




def run() -> None:
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    scene_cfg = DatasetSceneCfg(num_envs=1, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    (OUTPUT_DIR / "rgb").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "instance_seg").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "instance_seg_preview").mkdir(parents=True, exist_ok=True)
    wheel_paths, wheel_spacings = create_wheels(scene, spacings)
    ####### DEBUG #########

    import omni.usd
    from pxr import UsdGeom, Usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/Objects/Wheel_0")
    print("valid/loaded/active:", prim.IsValid(), prim.IsLoaded(), prim.IsActive())

    # Force load + visibility
    prim.Load()
    img = UsdGeom.Imageable(prim)
    print("visibility:", img.ComputeVisibility())
    img.MakeVisible()

    # Count renderable meshes
    mesh_count = 0
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            mesh_count += 1
    print("mesh_count:", mesh_count)

    # World bbox (if this is empty/zero, there’s likely no geometry)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default","render","proxy","guide"], useExtentsHint=True)
    bbox = bbox_cache.ComputeWorldBound(prim).GetRange()
    print("world_bbox_min/max:", bbox.GetMin(), bbox.GetMax())


    #######################
    vehicle = PseudoVehicle(
        wheel_paths=wheel_paths,
        v_mps=14.0,
        wheel_radius_m=0.45,  # ggf. an dein Wheel anpassen
        spin_axis="x",
    )
    vehicle.mount_quat_wxyz = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), -np.pi/2)
    vehicle.wheel_offsets = wheel_spacings
    vehicle.base_pos = np.array([0.0, 0.0, vehicle.wheel_radius_m], dtype=np.float64)
    vehicle.bind()
    vehicle.step(0.0)
    #vehicle.base_pos = np.array([0.0, 0.0, vehicle.wheel_radius_m], dtype=np.float64)

    frames = 500  # Beispiel
    for frame_idx in range(frames):
        if not simulation_app.is_running():
            break

        move_vehicles(vehicle, sim.get_physics_dt())

        scene.write_data_to_sim()
        sim.step()
        sim.render()
        scene.update(sim.get_physics_dt())

        cam = scene["camera"]
        rgb = cam.data.output["rgb"]  # typischerweise (H,W,4) oder Tensor
        seg = cam.data.output["instance_segmentation_fast"]

        # Tensor (cuda) -> numpy (cpu) FIRST
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        if hasattr(seg, "detach"):
            seg = seg.detach().cpu().numpy()

        # Now it’s safe to treat as numpy and drop batch/env dims
        rgb = np.asarray(rgb)
        seg = np.asarray(seg)

        while rgb.ndim > 3:
            rgb = rgb[0]
        while seg.ndim > 3:
            seg = seg[0]

        # ggf. Alpha droppen
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        _save_rgb_png(rgb, OUTPUT_DIR / "rgb" / f"{frame_idx:05d}.png")
        _save_instance_seg(
            seg,
            OUTPUT_DIR / "instance_seg" / f"{frame_idx:05d}.npy",
            preview_png_path=OUTPUT_DIR / "instance_seg_preview" / f"{frame_idx:05d}.png",
        )

    simulation_app.close()


if __name__ == "__main__":
    run()
