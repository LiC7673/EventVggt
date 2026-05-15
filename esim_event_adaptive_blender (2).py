import bpy
import os
import json
import math
import shutil
import tempfile
import traceback

import numpy as np

try:
    import h5py
except Exception as e:
    h5py = None
    H5PY_IMPORT_ERROR = e
else:
    H5PY_IMPORT_ERROR = None

# ============================================================
# 路径配置：保持和你的 LDR.py 一样的路径 / 思路
# ============================================================
OBJ_DIR = r"F:/TreeOBJ/stls"
HDRI_DIR = r"E:/dataSet/myblendevent/hdr_dataset"
OUTPUT_DIR = r"F:/TreeOBJ/reflective_raw"
LOG_FILE = r"F:/TreeOBJ/skipped_esim_event_dense.txt"

# ============================================================
# 基本渲染参数：与 LDR.py 对齐
# ============================================================
RESOLUTION_X = 640
RESOLUTION_Y = 480
FRAME_COUNT = 120
FPS = 120

# ============================================================
# 输出设置
# ============================================================
EVENT_SUBDIR = "cur_best_event"
EVENT_H5_NAME = "events.h5"

# 根目录下直接保存：h5_file["events"] -> shape=(N,4), columns=[t, x, y, p]
# 兼容你的 load_event_slice(path, start_idx, end_idx)
TIME_UNIT = "sec"

# ============================================================
# ESIM-style 事件参数：致密化但保持合理
# ============================================================
USE_LOG_IMAGE = True
LOG_EPS = 1e-6

# 比 0.18 更适合你的快速渲染 / 实物纹理场景。
# 如果事件仍少，先试 0.05；如果太噪，试 0.09。
CP = 0.07
CM = 0.07

# per-pixel threshold mismatch，更像真实 event camera。
ENABLE_THRESHOLD_MISMATCH = True
THRESHOLD_MISMATCH_STD = 0.20
THRESHOLD_MIN = 0.025
THRESHOLD_MAX = 0.25
RANDOM_SEED = 202405

# 不应期。一般先设 0；如果事件过密再设 50_000~200_000 ns。
REFRACTORY_PERIOD_NS = 0

# ============================================================
# 传感器噪声：用于让模拟 event 接近真实相机密度
# ============================================================
ENABLE_BACKGROUND_ACTIVITY = True
BACKGROUND_ACTIVITY_RATE = 0.02
# 单位 events / pixel / second。
# 640x480 下 0.02 约 6144 events/s。
# 可调范围：0.005 ~ 0.05。

ENABLE_BRIGHTNESS_NOISE = True
BRIGHT_NOISE_RATE = 0.04
# 亮度相关背景活动，HDR / 高亮区域更容易出噪声。
# 可调范围：0.01 ~ 0.10。

# 少量 hot pixels，更接近真实相机。默认很小。
ENABLE_HOT_PIXELS = True
HOT_PIXEL_RATIO = 0.00005
HOT_PIXEL_RATE = 1.0
# HOT_PIXEL_RATIO=0.00005 对 640x480 大约 15 个 hot pixels。
# HOT_PIXEL_RATE 单位 events / hot_pixel / second。

# ============================================================
# 渲染后端与材质策略
# ============================================================
# 事件分支默认用 EEVEE：只渲 anchor frame + 少量 midpoint。
# RGB/LDR 仍由你原脚本用 Cycles / Standard 输出。
EVENT_RENDER_BACKEND = "EEVEE"  # "EEVEE" or "CYCLES"
CYCLES_SAMPLES = 1

# 关键：默认保留 metadata 中的真实材质，而不是纯灰 proxy。
USE_METADATA_MATERIAL = True

# 为了事件更密，可对真实材质做轻微 event-friendly 调整：
# - 保留颜色/金属/roughness/coat/anisotropic 等 metadata 信息
# - 只对极端光滑材质加一点最小 roughness，避免全黑/全白镜面导致事件过少或 EEVEE 不稳定
ENABLE_EVENT_MATERIAL_STABILIZATION = True
MIN_EVENT_ROUGHNESS = 0.08

# 如果真实材质仍然太干净，可以额外叠加非常弱的 procedural albedo/noise。
# 默认关闭，避免偏离 metadata。事件太稀时可打开。
ADD_SUBTLE_PROCEDURAL_TEXTURE = False
SUBTLE_TEXTURE_STRENGTH = 0.25
SUBTLE_TEXTURE_SCALE = 65.0

# ============================================================
# 局部 B：局部 midpoint refine
# ============================================================
ENABLE_LOCAL_REFINEMENT = True

# 更积极一点，让高变化 / 高梯度区域补 midpoint，增强非线性 HDR/高光事件。
REFINE_DELTA_RATIO = 0.8
REFINE_GRAD_THRESHOLD = 0.015
REFINE_MIN_PIXEL_RATIO = 0.0003
REFINE_DILATE_MASK = True

# ============================================================
# 相机微振动：增加真实感与事件数
# ============================================================
ENABLE_CAMERA_JITTER = True
JITTER_ROT_DEG_SIN = 0.06
JITTER_ROT_DEG_RANDOM = 0.012
CP = 0.05
CM = 0.05
LOG_EPS = 1e-6

ENABLE_THRESHOLD_MISMATCH = True
THRESHOLD_MISMATCH_STD = 0.30
THRESHOLD_MIN = 0.015
THRESHOLD_MAX = 0.20

ADD_SUBTLE_PROCEDURAL_TEXTURE = True
SUBTLE_TEXTURE_STRENGTH = 0.25
SUBTLE_TEXTURE_SCALE = 65.0

# ENABLE_LOCAL_REFINEMENT = True
# REFINE_DELTA_RATIO = 0.5
# REFINE_GRAD_THRESHOLD = 0.008
# REFINE_MIN_PIXEL_RATIO = 0.0001

# ENABLE_BACKGROUND_ACTIVITY = True
# BACKGROUND_ACTIVITY_RATE = 0.05

# ENABLE_BRIGHTNESS_NOISE = True
# BRIGHT_NOISE_RATE = 0.08

# ENABLE_HOT_PIXELS = True
# HOT_PIXEL_RATIO = 0.0001
# HOT_PIXEL_RATE = 2.0

# ENABLE_CAMERA_JITTER = True
# JITTER_ROT_DEG_SIN = 0.10
# JITTER_ROT_DEG_RANDOM = 0.02

# MIN_EVENT_ROUGHNESS = 0.05
# ============================================================
# Render Result fallback
# ============================================================
KEEP_TMP_EXR = False

# ============================================================
# 日志 / 文件工具
# ============================================================

def log_error(msg):
    print(f"[Error] {msg}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    except Exception:
        pass

def find_pose_json(scene_folder_path):
    """
    在 scene 目录和 HDR 子目录里找 trans / transforms / cameras pose JSON。
    """
    search_dirs = [
        scene_folder_path,
        os.path.join(scene_folder_path, "HDR"),
    ]

    for d in search_dirs:
        for name in POSE_JSON_CANDIDATES:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p

    return None


def as_matrix4x4(mat):
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape == (4, 4):
        return mathutils.Matrix(arr.tolist())
    if arr.size == 16:
        return mathutils.Matrix(arr.reshape(4, 4).tolist())
    raise ValueError(f"Invalid pose matrix shape: {arr.shape}")


def convert_pose_matrix(mat):
    """
    返回 Blender camera.matrix_world。

    POSE_CONVENTION='blender':
        JSON 里就是 Blender camera-to-world matrix_world。

    POSE_CONVENTION='opencv_c2w':
        JSON 是 OpenCV camera-to-world，camera 坐标 x右 y下 z前。
        Blender camera 坐标 x右 y上 -z前，所以右乘 diag(1,-1,-1,1)。
    """
    M = as_matrix4x4(mat)

    if POSE_CONVENTION == "blender":
        return M

    if POSE_CONVENTION == "opencv_c2w":
        cv_to_blender = mathutils.Matrix((
            (1,  0,  0, 0),
            (0, -1,  0, 0),
            (0,  0, -1, 0),
            (0,  0,  0, 1),
        ))
        return M @ cv_to_blender

    raise ValueError(f"Unsupported POSE_CONVENTION: {POSE_CONVENTION}")


def get_frame_index_from_record(record, default_idx):
    """
    尝试从 file_path / frame / index 里解析帧号。
    返回 Blender frame index，通常从 1 开始。
    """
    for key in ["frame", "frame_id", "idx", "index"]:
        if key in record:
            try:
                v = int(record[key])
                return v if v >= 1 else v + 1
            except Exception:
                pass

    fp = record.get("file_path", "") or record.get("path", "") or record.get("name", "")
    if fp:
        base = os.path.basename(fp)
        stem = os.path.splitext(base)[0]
        digits = "".join([c if c.isdigit() else " " for c in stem]).split()
        if digits:
            v = int(digits[-1])
            return v if v >= 1 else v + 1

    return default_idx

def reconstruct_camera_anim(target_obj):
    # 完全复制原脚本的相机逻辑
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    empty = bpy.data.objects.new("Empty_Center", None)
    bpy.context.collection.objects.link(empty)
    empty.location = target_obj.location
    
    cam_obj.parent = empty
    cam_obj.location = (0, -2.0, 1.0)
    
    constraint = cam_obj.constraints.new(type='TRACK_TO')
    constraint.target = target_obj
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    empty.rotation_euler = (0, 0, 0)
    empty.keyframe_insert(data_path="rotation_euler", frame=1)
    empty.rotation_euler = (0, 0, math.radians(360))
    empty.keyframe_insert(data_path="rotation_euler", frame=FRAME_COUNT + 1)
    
    for fcurve in empty.animation_data.action.fcurves:
        for kf in fcurve.keyframe_points: kf.interpolation = 'LINEAR'

def parse_pose_json(pose_json_path):
    """
    兼容几种常见格式：

    1) NeRF transforms.json:
       {
         "frames": [
           {"file_path": "...0001", "transform_matrix": [[...]]},
           ...
         ]
       }

    2) cameras.json / poses dict:
       {
         "0001": {"matrix_world": [[...]]},
         "0002": {"matrix_world": [[...]]}
       }

    3) list:
       [
         [[4x4]],
         [[4x4]]
       ]

    返回:
       dict: frame_idx -> Blender matrix_world
    """
    with open(pose_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame_to_mat = {}

    # 格式 1: {"frames": [...]}
    if isinstance(data, dict) and "frames" in data and isinstance(data["frames"], list):
        for i, rec in enumerate(data["frames"], start=1):
            if not isinstance(rec, dict):
                continue

            mat = None
            for k in ["transform_matrix", "matrix_world", "c2w", "pose", "camera_to_world"]:
                if k in rec:
                    mat = rec[k]
                    break

            if mat is None:
                continue

            frame_idx = get_frame_index_from_record(rec, i)
            frame_to_mat[frame_idx] = convert_pose_matrix(mat)

        return frame_to_mat

    # 格式 2: list of 4x4 matrices
    if isinstance(data, list):
        for i, mat in enumerate(data, start=1):
            frame_to_mat[i] = convert_pose_matrix(mat)
        return frame_to_mat

    # 格式 3: dict by frame key
    if isinstance(data, dict):
        # 有些文件可能把内参和 poses 分开存在 poses/cameras
        if "poses" in data and isinstance(data["poses"], (list, dict)):
            pose_data = data["poses"]
            if isinstance(pose_data, list):
                for i, mat in enumerate(pose_data, start=1):
                    frame_to_mat[i] = convert_pose_matrix(mat)
                return frame_to_mat
            if isinstance(pose_data, dict):
                data = pose_data

        for key, rec in data.items():
            if key in ["camera_angle_x", "fl_x", "fl_y", "cx", "cy", "w", "h", "intrinsics"]:
                continue

            try:
                frame_idx = int(key)
                if frame_idx == 0:
                    frame_idx = 1
            except Exception:
                # 例如 frame_0001
                digits = "".join([c if c.isdigit() else " " for c in str(key)]).split()
                if not digits:
                    continue
                frame_idx = int(digits[-1])
                if frame_idx == 0:
                    frame_idx = 1

            mat = None
            if isinstance(rec, dict):
                for k in ["matrix_world", "transform_matrix", "c2w", "pose", "camera_to_world"]:
                    if k in rec:
                        mat = rec[k]
                        break
            else:
                mat = rec

            if mat is None:
                continue

            frame_to_mat[frame_idx] = convert_pose_matrix(mat)

        return frame_to_mat

    raise ValueError(f"Unsupported pose json format: {pose_json_path}")


def apply_camera_intrinsics_from_json(cam_data, pose_json_path, meta=None):
    """
    尽量从 trans json 或 metadata 中恢复相机内参。
    没有的话用默认 lens。
    """
    cam_data.lens = CAMERA_LENS_MM
    cam_data.sensor_width = CAMERA_SENSOR_WIDTH

    try:
        with open(pose_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    # NeRF-style camera_angle_x
    if isinstance(data, dict):
        w = data.get("w", RESOLUTION_X)
        camera_angle_x = data.get("camera_angle_x", None)
        if camera_angle_x is not None:
            # focal pixel -> lens mm
            fl_px = 0.5 * float(w) / math.tan(0.5 * float(camera_angle_x))
            cam_data.sensor_width = CAMERA_SENSOR_WIDTH
            cam_data.lens = fl_px * cam_data.sensor_width / float(w)
            return

        # fl_x style
        fl_x = data.get("fl_x", None)
        if fl_x is not None:
            w = float(data.get("w", RESOLUTION_X))
            cam_data.sensor_width = CAMERA_SENSOR_WIDTH
            cam_data.lens = float(fl_x) * cam_data.sensor_width / w
            return

    # metadata 里如果有 intrinsics，也可以读
    if isinstance(meta, dict):
        intr = meta.get("intrinsics", None) or meta.get("camera", None)
        if isinstance(intr, dict):
            fl_x = intr.get("fl_x", None) or intr.get("fx", None)
            w = float(intr.get("w", RESOLUTION_X))
            if fl_x is not None:
                cam_data.sensor_width = CAMERA_SENSOR_WIDTH
                cam_data.lens = float(fl_x) * cam_data.sensor_width / w


def reconstruct_camera_from_pose_json(scene_folder_path, meta=None, fallback_target_obj=None):
    """
    用 trans / transforms JSON 中的 pose 创建相机动画。
    这才会和已有 RGB/LDR/HDR 的真实位姿对齐。
    """
    pose_json_path = find_pose_json(scene_folder_path)
    if pose_json_path is None:
        raise FileNotFoundError(
            f"No pose json found in {scene_folder_path} or HDR subfolder. "
            f"Tried: {POSE_JSON_CANDIDATES}"
        )

    frame_to_mat = parse_pose_json(pose_json_path)
    if not frame_to_mat:
        raise ValueError(f"No valid camera poses parsed from: {pose_json_path}")

    cam_data = bpy.data.cameras.new("Camera")
    apply_camera_intrinsics_from_json(cam_data, pose_json_path, meta=meta)

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # 不要 parent empty，不要 track_to constraint，否则会覆盖真实 pose。
    for frame_idx in sorted(frame_to_mat.keys()):
        if frame_idx < 1 or frame_idx > FRAME_COUNT:
            continue

        cam_obj.matrix_world = frame_to_mat[frame_idx]

        # matrix_world 直接 keyframe 通常不如分解 location/quaternion 稳
        loc, rot, scale = cam_obj.matrix_world.decompose()
        cam_obj.location = loc
        cam_obj.rotation_mode = "QUATERNION"
        cam_obj.rotation_quaternion = rot

        cam_obj.keyframe_insert(data_path="location", frame=frame_idx)
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

    if cam_obj.animation_data and cam_obj.animation_data.action:
        for fcurve in cam_obj.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = "LINEAR"

    print(f"[Pose] Loaded {len(frame_to_mat)} camera poses from: {pose_json_path}")
    return cam_obj
def get_obj_map(folder):
    mapping = {}
    valid_exts = ('.obj', '.stl', '.glb', '.gltf', '.ply')
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(valid_exts):
                full_path = os.path.join(root, f)
                mapping[f] = full_path
                mapping[os.path.splitext(f)[0]] = full_path
    return mapping


def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        try:
            bpy.data.images.remove(block)
        except Exception:
            pass
    for block in list(bpy.data.cameras):
        bpy.data.cameras.remove(block)
    for block in list(bpy.data.lights):
        try:
            bpy.data.lights.remove(block)
        except Exception:
            pass


# ============================================================
# Blender 渲染设置
# ============================================================

def setup_renderer_event():
    scene = bpy.context.scene

    if EVENT_RENDER_BACKEND == "EEVEE":
        try:
            scene.render.engine = 'BLENDER_EEVEE'
        except Exception:
            try:
                scene.render.engine = 'BLENDER_EEVEE_NEXT'
            except Exception:
                scene.render.engine = 'CYCLES'

        # EEVEE 下尽量快，但保留 basic specular/reflection response。
        try:
            scene.eevee.taa_render_samples = 1
        except Exception:
            pass
        try:
            scene.eevee.taa_samples = 1
        except Exception:
            pass
        try:
            scene.eevee.use_gtao = False
        except Exception:
            pass
        try:
            scene.eevee.use_bloom = False
        except Exception:
            pass
        # SSR 可保留一定反射变化；若崩溃/慢，设 False。
        try:
            scene.eevee.use_ssr = True
        except Exception:
            pass
    else:
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'
            prefs.get_devices()
            for device in prefs.devices:
                device.use = True
        except Exception:
            pass

        scene.cycles.samples = CYCLES_SAMPLES
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 2
        scene.cycles.glossy_bounces = 1
        scene.cycles.transparent_max_bounces = 2
        scene.cycles.transmission_max_bounces = 2

    try:
        bpy.context.preferences.viewport.use_gpu_subdivision = False
    except Exception:
        pass

    scene.render.use_persistent_data = True
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.fps = FPS

    # 事件分支一定要 Raw / linear，不使用 LDR 曝光/tone mapping。
    try:
        scene.view_settings.view_transform = 'Raw'
    except Exception:
        scene.view_settings.view_transform = 'Standard'
    try:
        scene.view_settings.look = 'None'
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # fallback EXR
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '32'


# ============================================================
# 场景重建：基于你的 LDR.py 思路
# ============================================================

def normalize_object_raw(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)

    max_dim = max(obj.dimensions)
    if max_dim > 0:
        scale_factor = 1.0 / max_dim
        obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(scale=True, location=True)

    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(45))
    except AttributeError:
        bpy.ops.object.shade_smooth()
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.radians(45)


def reconstruct_material_from_metadata(obj, mat_data):
    """
    基于你 LDR.py 中 reconstruct_material 的思路：
    从 metadata.json 恢复真实材质信息。
    这里增强了字段兼容性，并为 event 分支加入轻微稳定化。
    """
    mat_name = mat_data.get("name", "metadata_material")
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    if bsdf is None:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        return mat

    def safe_set(input_names, value):
        if not isinstance(input_names, list):
            input_names = [input_names]
        for name in input_names:
            if name in bsdf.inputs:
                try:
                    bsdf.inputs[name].default_value = value
                    return True
                except Exception:
                    return False
        return False

    color = mat_data.get("color", (0.8, 0.8, 0.8, 1.0))
    safe_set(["Base Color", "Color"], color)

    metallic = mat_data.get("metallic", 0.0)
    if mat_data.get("type", "") == "metallic":
        metallic = 1.0
    safe_set("Metallic", metallic)

    roughness = mat_data.get("roughness", 0.5)
    if ENABLE_EVENT_MATERIAL_STABILIZATION:
        roughness = max(float(roughness), MIN_EVENT_ROUGHNESS)
    safe_set("Roughness", roughness)

    safe_set(["Coat Weight", "Clearcoat"], mat_data.get("coat_weight", mat_data.get("clearcoat", 0.0)))
    safe_set(["Coat Roughness", "Clearcoat Roughness"], mat_data.get("coat_roughness", mat_data.get("clearcoat_roughness", 0.03)))
    safe_set(["Anisotropic", "Anisotropic Rotation"], mat_data.get("anisotropic", 0.0))

    if "ior" in mat_data:
        safe_set("IOR", mat_data.get("ior", 1.45))
    if "transmission" in mat_data:
        safe_set(["Transmission Weight", "Transmission"], mat_data.get("transmission", 0.0))
    if "alpha" in mat_data:
        safe_set("Alpha", mat_data.get("alpha", 1.0))

    # 可选：在 metadata 材质基础上叠加非常弱的 procedural texture，增加真实微纹理事件。
    # 默认关闭。如果打开，它不是纯 proxy，而是对真实颜色做小幅扰动。
    if ADD_SUBTLE_PROCEDURAL_TEXTURE:
        base_color = color
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = SUBTLE_TEXTURE_SCALE
        noise.inputs["Detail"].default_value = 10.0
        noise.inputs["Roughness"].default_value = 0.55

        ramp = nodes.new(type="ShaderNodeValToRGB")
        lo = max(0.0, 0.5 - 0.5 * SUBTLE_TEXTURE_STRENGTH)
        hi = min(1.0, 0.5 + 0.5 * SUBTLE_TEXTURE_STRENGTH)
        ramp.color_ramp.elements[0].position = 0.25
        ramp.color_ramp.elements[0].color = (lo * base_color[0], lo * base_color[1], lo * base_color[2], 1.0)
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = (hi * base_color[0], hi * base_color[1], hi * base_color[2], 1.0)

        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return mat


def reconstruct_hdri(hdri_filename):
    hdri_path = os.path.join(HDRI_DIR, hdri_filename)
    if not os.path.exists(hdri_path):
        print(f"Warning: HDRI not found {hdri_path}")
        return

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    world.node_tree.nodes.clear()

    node_out = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    node_bg = world.node_tree.nodes.new(type='ShaderNodeBackground')
    node_env = world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')

    node_env.image = bpy.data.images.load(hdri_path)
    node_bg.inputs['Strength'].default_value = 1.5

    links = world.node_tree.links
    links.new(node_env.outputs['Color'], node_bg.inputs['Color'])
    links.new(node_bg.outputs['Background'], node_out.inputs['Surface'])





def import_model(obj_path):
    ext = os.path.splitext(obj_path)[1].lower()
    if ext == '.obj':
        if hasattr(bpy.ops.wm, 'obj_import'):
            bpy.ops.wm.obj_import(filepath=obj_path)
        else:
            bpy.ops.import_scene.obj(filepath=obj_path)
    elif ext == '.stl':
        if hasattr(bpy.ops.wm, 'stl_import'):
            bpy.ops.wm.stl_import(filepath=obj_path)
        else:
            bpy.ops.import_mesh.stl(filepath=obj_path)
    elif ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=obj_path)
    elif ext == '.ply':
        if hasattr(bpy.ops.wm, 'ply_import'):
            bpy.ops.wm.ply_import(filepath=obj_path)
        else:
            bpy.ops.import_mesh.ply(filepath=obj_path)
    else:
        raise RuntimeError(f"Unsupported model format: {ext}")


# ============================================================
# 渲染 log-luma：anchor frame + local midpoint
# ============================================================

class LumaRenderer:
    def __init__(self, tmp_dir=None):
        if tmp_dir is None:
            self.tmp_dir = tempfile.mkdtemp(prefix="blender_dense_event_")
            self.own_tmp_dir = True
        else:
            self.tmp_dir = tmp_dir
            os.makedirs(self.tmp_dir, exist_ok=True)
            self.own_tmp_dir = False

        self.cache = {}
        self.render_count = 0

    def cleanup(self):
        self.cache.clear()
        if self.own_tmp_dir and (not KEEP_TMP_EXR):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def clear_cache_except(self, keep_keys):
        keep = set(keep_keys)
        for k in list(self.cache.keys()):
            if k not in keep:
                del self.cache[k]

    def _set_frame_float(self, frame_float):
        frame_float = max(1.0, min(float(frame_float), float(FRAME_COUNT)))
        base_frame = int(math.floor(frame_float))
        subframe = frame_float - base_frame
        if subframe < 0.0:
            subframe = 0.0
        if subframe >= 1.0:
            subframe = 0.999999
        bpy.context.scene.frame_set(base_frame, subframe=subframe)

    def _pixels_to_log_luma(self, pixels_flat, width, height):
        expected = int(width * height * 4)
        if width <= 0 or height <= 0 or len(pixels_flat) != expected:
            raise RuntimeError(
                f"Invalid pixel buffer: size=({width},{height}), "
                f"pixel_count={len(pixels_flat)}, expected={expected}"
            )

        pixels = np.array(pixels_flat[:], dtype=np.float32).reshape(height, width, 4)
        rgb = pixels[:, :, :3]
        luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        luma = np.clip(luma, 1e-6, None)

        if USE_LOG_IMAGE:
            luma = np.log(LOG_EPS + luma)

        return luma.astype(np.float32)

    def render(self, frame_float):
        key = round(float(frame_float), 6)
        if key in self.cache:
            return self.cache[key]

        self._set_frame_float(frame_float)
        bpy.ops.render.render(write_still=False)

        rr = bpy.data.images.get("Render Result")
        if rr is not None:
            try:
                w, h = rr.size
                if w > 0 and h > 0 and len(rr.pixels) == int(w * h * 4):
                    L = self._pixels_to_log_luma(rr.pixels, w, h)
                    self.cache[key] = L
                    self.render_count += 1
                    return L
            except Exception as e:
                print(f"[Warn] Render Result read failed at frame {frame_float}: {e}")

        # fallback：EXR 回读
        if rr is None:
            raise RuntimeError("Render Result is None after rendering")

        exr_path = os.path.join(self.tmp_dir, f"tmp_{self.render_count:08d}.exr")
        rr.save_render(filepath=exr_path, scene=bpy.context.scene)
        if not os.path.exists(exr_path):
            raise RuntimeError(f"Fallback EXR not created: {exr_path}")

        img = bpy.data.images.load(exr_path, check_existing=False)
        try:
            w, h = img.size
            if w <= 0 or h <= 0 or len(img.pixels) != int(w * h * 4):
                raise RuntimeError(
                    f"Invalid fallback EXR pixels: size=({w},{h}), "
                    f"pixel_count={len(img.pixels)}, expected={int(w*h*4)}"
                )
            L = self._pixels_to_log_luma(img.pixels, w, h)
        finally:
            bpy.data.images.remove(img)
            if not KEEP_TMP_EXR:
                try:
                    os.remove(exr_path)
                except Exception:
                    pass

        self.cache[key] = L
        self.render_count += 1
        return L


# ============================================================
# Event state / writer
# ============================================================

class EventState:
    def __init__(self, first_log_img):
        h, w = first_log_img.shape
        self.ref_values = first_log_img.copy()
        self.last_event_timestamp = np.zeros((h, w), dtype=np.int64)
        self.height = h
        self.width = w

        rng = np.random.default_rng(RANDOM_SEED)
        if ENABLE_THRESHOLD_MISMATCH:
            cp = CP * (1.0 + rng.normal(0.0, THRESHOLD_MISMATCH_STD, size=(h, w)))
            cm = CM * (1.0 + rng.normal(0.0, THRESHOLD_MISMATCH_STD, size=(h, w)))
            self.cp_map = np.clip(cp, THRESHOLD_MIN, THRESHOLD_MAX).astype(np.float32)
            self.cm_map = np.clip(cm, THRESHOLD_MIN, THRESHOLD_MAX).astype(np.float32)
        else:
            self.cp_map = np.full((h, w), CP, dtype=np.float32)
            self.cm_map = np.full((h, w), CM, dtype=np.float32)

        if ENABLE_HOT_PIXELS:
            n_hot = int(round(HOT_PIXEL_RATIO * h * w))
            hot = np.zeros((h, w), dtype=bool)
            if n_hot > 0:
                idx = rng.choice(h * w, size=n_hot, replace=False)
                hot.reshape(-1)[idx] = True
            self.hot_pixel_mask = hot
        else:
            self.hot_pixel_mask = np.zeros((h, w), dtype=bool)


class EventArrayH5Writer:
    def __init__(self, h5_path, scene_id):
        if h5py is None:
            raise RuntimeError(
                "h5py is not installed in Blender Python. "
                f"Original import error: {H5PY_IMPORT_ERROR}"
            )

        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        self.f = h5py.File(h5_path, "w")
        self.ds = self.f.create_dataset(
            "events",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype=np.float32,
            chunks=(65536, 4),
            compression="gzip",
        )
        self.count = 0

        # 不影响你的 loader，但方便复现实验。
        self.ds.attrs["format"] = "events[N,4] = [t, x, y, p]"
        self.ds.attrs["time_unit"] = TIME_UNIT
        self.ds.attrs["scene_id"] = scene_id
        self.ds.attrs["width"] = RESOLUTION_X
        self.ds.attrs["height"] = RESOLUTION_Y
        self.ds.attrs["fps"] = FPS
        self.ds.attrs["frame_count"] = FRAME_COUNT
        self.ds.attrs["Cp"] = CP
        self.ds.attrs["Cm"] = CM
        self.ds.attrs["threshold_mismatch"] = int(ENABLE_THRESHOLD_MISMATCH)
        self.ds.attrs["threshold_mismatch_std"] = THRESHOLD_MISMATCH_STD
        self.ds.attrs["background_activity"] = int(ENABLE_BACKGROUND_ACTIVITY)
        self.ds.attrs["background_activity_rate"] = BACKGROUND_ACTIVITY_RATE
        self.ds.attrs["brightness_noise"] = int(ENABLE_BRIGHTNESS_NOISE)
        self.ds.attrs["bright_noise_rate"] = BRIGHT_NOISE_RATE
        self.ds.attrs["hot_pixels"] = int(ENABLE_HOT_PIXELS)
        self.ds.attrs["hot_pixel_ratio"] = HOT_PIXEL_RATIO
        self.ds.attrs["hot_pixel_rate"] = HOT_PIXEL_RATE
        self.ds.attrs["metadata_material"] = int(USE_METADATA_MATERIAL)
        self.ds.attrs["event_material_stabilization"] = int(ENABLE_EVENT_MATERIAL_STABILIZATION)
        self.ds.attrs["local_refinement"] = int(ENABLE_LOCAL_REFINEMENT)
        self.ds.attrs["camera_jitter"] = int(ENABLE_CAMERA_JITTER)

    def append(self, events_list):
        if len(events_list) == 0:
            self.f.flush()
            return

        arr = np.asarray(events_list, dtype=np.float32)
        order = np.argsort(arr[:, 0])
        arr = arr[order]

        new_size = self.count + arr.shape[0]
        self.ds.resize((new_size, 4))
        self.ds[self.count:new_size, :] = arr
        self.count = new_size
        self.f.flush()

    def close(self):
        if self.f is None:
            return
        self.ds.attrs["event_count"] = int(self.count)
        self.f.flush()
        self.f.close()
        self.f = None


# ============================================================
# Core event functions
# ============================================================

def compute_gradient_magnitude(img):
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)

    gx[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    gx[:, 0] = img[:, 1] - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]

    gy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
    gy[0, :] = img[1, :] - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]

    return np.sqrt(gx * gx + gy * gy)


def dilate_mask_3x3(mask):
    out = mask.copy()
    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    for dy, dx in shifts:
        rolled = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        out |= rolled
    return out


def compute_local_refine_mask(L0, L1):
    delta = np.abs(L1 - L0)
    grad = compute_gradient_magnitude(L0)
    mask = (delta > REFINE_DELTA_RATIO * min(CP, CM)) & (grad > REFINE_GRAD_THRESHOLD)

    if REFINE_DILATE_MASK:
        mask = dilate_mask_3x3(mask)

    if float(mask.mean()) < REFINE_MIN_PIXEL_RATIO:
        return np.zeros_like(mask, dtype=bool)
    return mask


def emit_events_for_segment(L_start, L_end, t0_ns, t1_ns, state, events_out, mask=None):
    if t1_ns <= t0_ns:
        return

    h, w = L_start.shape
    if mask is None:
        mask = np.ones((h, w), dtype=bool)

    tolerance = 1e-6
    delta_t_ns = float(t1_ns - t0_ns)

    ref = state.ref_values
    last_ts = state.last_event_timestamp
    diff = L_end - L_start

    pos_candidate = mask & (diff > tolerance)
    neg_candidate = mask & (diff < -tolerance)

    if np.any(pos_candidate):
        # 用全局 CP 粗筛，再在像素 loop 中使用 cp_map 精确算。
        pos_check = (L_end - ref) > np.minimum(state.cp_map, CP)
        pos_idx = np.where(pos_candidate & pos_check)
    else:
        pos_idx = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    if np.any(neg_candidate):
        neg_check = (ref - L_end) > np.minimum(state.cm_map, CM)
        neg_idx = np.where(neg_candidate & neg_check)
    else:
        neg_idx = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    for y, x in zip(pos_idx[0], pos_idx[1]):
        it = float(L_start[y, x])
        itdt = float(L_end[y, x])
        prev = float(ref[y, x])
        C = float(state.cp_map[y, x])

        if itdt <= it:
            continue

        m_start = int(math.floor((it - prev) / C)) + 1
        m_end = int(math.floor((itdt - prev) / C))
        m_start = max(m_start, 1)
        if m_end < m_start:
            continue

        last_cross = prev
        for m in range(m_start, m_end + 1):
            Lc = prev + m * C
            alpha = (Lc - it) / (itdt - it)
            alpha = min(max(alpha, 0.0), 1.0)
            t_event_ns = int(round(t0_ns + alpha * delta_t_ns))

            last_stamp = int(last_ts[y, x])
            if t_event_ns < last_stamp:
                t_event_ns = last_stamp
            dt_last = t_event_ns - last_stamp

            if last_stamp == 0 or dt_last >= REFRACTORY_PERIOD_NS:
                t_out = np.float32(t_event_ns * 1e-9 if TIME_UNIT == "sec" else t_event_ns)
                events_out.append((t_out, np.float32(x), np.float32(y), np.float32(1.0)))
                last_ts[y, x] = t_event_ns

            last_cross = Lc

        ref[y, x] = last_cross

    for y, x in zip(neg_idx[0], neg_idx[1]):
        it = float(L_start[y, x])
        itdt = float(L_end[y, x])
        prev = float(ref[y, x])
        C = float(state.cm_map[y, x])

        if itdt >= it:
            continue

        m_start = int(math.floor((prev - it) / C)) + 1
        m_end = int(math.floor((prev - itdt) / C))
        m_start = max(m_start, 1)
        if m_end < m_start:
            continue

        last_cross = prev
        for m in range(m_start, m_end + 1):
            Lc = prev - m * C
            alpha = (Lc - it) / (itdt - it)
            alpha = min(max(alpha, 0.0), 1.0)
            t_event_ns = int(round(t0_ns + alpha * delta_t_ns))

            last_stamp = int(last_ts[y, x])
            if t_event_ns < last_stamp:
                t_event_ns = last_stamp
            dt_last = t_event_ns - last_stamp

            if last_stamp == 0 or dt_last >= REFRACTORY_PERIOD_NS:
                t_out = np.float32(t_event_ns * 1e-9 if TIME_UNIT == "sec" else t_event_ns)
                events_out.append((t_out, np.float32(x), np.float32(y), np.float32(-1.0)))
                last_ts[y, x] = t_event_ns

            last_cross = Lc

        ref[y, x] = last_cross


def generate_background_activity_events(t0_sec, t1_sec, width, height, rng):
    if not ENABLE_BACKGROUND_ACTIVITY:
        return []
    dt = max(0.0, t1_sec - t0_sec)
    expected = BACKGROUND_ACTIVITY_RATE * width * height * dt
    n = rng.poisson(expected)
    if n <= 0:
        return []

    xs = rng.integers(0, width, size=n)
    ys = rng.integers(0, height, size=n)
    ts = rng.uniform(t0_sec, t1_sec, size=n).astype(np.float32)
    ps = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    return [(ts[i], np.float32(xs[i]), np.float32(ys[i]), ps[i]) for i in range(n)]


def generate_brightness_noise_events(L0, t0_sec, t1_sec, rng):
    if not ENABLE_BRIGHTNESS_NOISE:
        return []
    dt = max(0.0, t1_sec - t0_sec)
    if dt <= 0:
        return []

    I = np.exp(L0).astype(np.float32)
    norm = np.percentile(I, 99.5) + 1e-6
    I = np.clip(I / norm, 0.0, 1.0)

    prob = BRIGHT_NOISE_RATE * I * dt
    rand = rng.random(I.shape)
    ys, xs = np.where(rand < prob)
    n = len(xs)
    if n <= 0:
        return []

    ts = rng.uniform(t0_sec, t1_sec, size=n).astype(np.float32)
    ps = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    return [(ts[i], np.float32(xs[i]), np.float32(ys[i]), ps[i]) for i in range(n)]


def generate_hot_pixel_events(state, t0_sec, t1_sec, rng):
    if not ENABLE_HOT_PIXELS or not np.any(state.hot_pixel_mask):
        return []
    dt = max(0.0, t1_sec - t0_sec)
    ys, xs = np.where(state.hot_pixel_mask)
    if len(xs) == 0:
        return []

    expected = HOT_PIXEL_RATE * dt
    out = []
    for y, x in zip(ys, xs):
        n = rng.poisson(expected)
        if n <= 0:
            continue
        ts = rng.uniform(t0_sec, t1_sec, size=n).astype(np.float32)
        ps = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
        for i in range(n):
            out.append((ts[i], np.float32(x), np.float32(y), ps[i]))
    return out


def simulate_interval_dense(frame_idx, L0, L1, renderer, state, rng):
    t0_sec = (frame_idx - 1) / FPS
    t1_sec = frame_idx / FPS
    t0_ns = int(round(t0_sec * 1e9))
    t1_ns = int(round(t1_sec * 1e9))
    tm_ns = int(round(0.5 * (t0_ns + t1_ns)))

    interval_events = []

    if ENABLE_LOCAL_REFINEMENT:
        refine_mask = compute_local_refine_mask(L0, L1)
    else:
        refine_mask = np.zeros_like(L0, dtype=bool)

    if np.any(refine_mask):
        base_mask = ~refine_mask
        if np.any(base_mask):
            emit_events_for_segment(L0, L1, t0_ns, t1_ns, state, interval_events, mask=base_mask)

        Lm = renderer.render(frame_idx + 0.5)
        emit_events_for_segment(L0, Lm, t0_ns, tm_ns, state, interval_events, mask=refine_mask)
        emit_events_for_segment(Lm, L1, tm_ns, t1_ns, state, interval_events, mask=refine_mask)
        print(f"    [LocalB] interval {frame_idx}/{FRAME_COUNT-1}, refine_ratio={float(refine_mask.mean()):.5f}")
    else:
        emit_events_for_segment(L0, L1, t0_ns, t1_ns, state, interval_events, mask=None)

    # 真实感致密化：加背景活动、亮度相关噪声、hot pixels。
    interval_events.extend(generate_background_activity_events(t0_sec, t1_sec, RESOLUTION_X, RESOLUTION_Y, rng))
    interval_events.extend(generate_brightness_noise_events(L0, t0_sec, t1_sec, rng))
    interval_events.extend(generate_hot_pixel_events(state, t0_sec, t1_sec, rng))

    return interval_events


# ============================================================
# 主生成流程
# ============================================================

def generate_events_dense(h5_path, scene_id):
    rng = np.random.default_rng(RANDOM_SEED)
    renderer = LumaRenderer()
    writer = None

    try:
        L_prev = renderer.render(1.0)
        state = EventState(L_prev)
        writer = EventArrayH5Writer(h5_path, scene_id)

        for frame_idx in range(1, FRAME_COUNT):
            L_next = renderer.render(float(frame_idx + 1))

            interval_events = simulate_interval_dense(
                frame_idx=frame_idx,
                L0=L_prev,
                L1=L_next,
                renderer=renderer,
                state=state,
                rng=rng,
            )

            writer.append(interval_events)

            delta = np.abs(L_next - L_prev)
            print(
                f"    [Flush] interval {frame_idx}/{FRAME_COUNT-1}, "
                f"interval_events={len(interval_events)}, total_events={writer.count}, "
                f"render_count={renderer.render_count}, "
                f"delta_mean={float(delta.mean()):.5f}, delta_max={float(delta.max()):.5f}, "
                f"delta>CP={float((delta > CP).mean()):.5f}"
            )

            interval_events.clear()

            # 保留下一帧亮度，删掉中间帧缓存，防止内存涨。
            keep_key = round(float(frame_idx + 1), 6)
            renderer.clear_cache_except({keep_key})
            L_prev = L_next

        writer.close()
        writer = None

    finally:
        if writer is not None:
            writer.close()
        renderer.cleanup()


# ============================================================
# 单场景处理
# ============================================================

def process_scene_folder(scene_folder_path, obj_mapping):
    json_path = os.path.join(scene_folder_path, "HDR", "metadata.json")
    if not os.path.exists(json_path):
        return

    scene_id = os.path.basename(scene_folder_path)
    event_dir = os.path.join(scene_folder_path, EVENT_SUBDIR)
    h5_path = os.path.join(event_dir, EVENT_H5_NAME)

    if os.path.exists(h5_path):
        print(f"Skipping {scene_id} (event h5 exists)")
        return

    print(f"--- 生成 Dense Event: {scene_id} ---")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        log_error(f"Metadata Error {scene_id}: {e}")
        return

    clean_scene()
    setup_renderer_event()

    obj_id = meta.get("object_id")
    obj_path = obj_mapping.get(obj_id)
    if not obj_path:
        log_error(f"Missing Model Source: {obj_id} (in scene {scene_id})")
        return

    try:
        import_model(obj_path)
    except Exception as e:
        log_error(f"Import Error {obj_id}: {e}")
        return

    if len(bpy.context.selected_objects) == 0:
        log_error(f"Import Error {obj_id}: no selected object after import")
        return

    imported_obj = bpy.context.selected_objects[0]

    try:
        normalize_object_raw(imported_obj)
        if USE_METADATA_MATERIAL:
            reconstruct_material_from_metadata(imported_obj, meta.get("material", {}))
        else:
            reconstruct_material_from_metadata(imported_obj, {})
        reconstruct_hdri(meta.get("hdri_file", ""))
        reconstruct_camera_anim(imported_obj)
    except Exception as e:
        log_error(f"Scene Reconstruct Error {scene_id}: {e}")
        log_error(traceback.format_exc())
        return

    try:
        generate_events_dense(h5_path, scene_id)
        print(f"  > Saved: {h5_path}")
    except Exception as e:
        log_error(f"Event Render Error {scene_id}: {e}")
        log_error(traceback.format_exc())


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    if h5py is None:
        print("[Fatal] h5py is not installed in Blender Python.")
        print(f"[Fatal] Original error: {H5PY_IMPORT_ERROR}")
        raise SystemExit(1)

    print("正在索引原始模型库 (OBJ_DIR)...")
    obj_map = get_obj_map(OBJ_DIR)

    if not os.path.exists(OUTPUT_DIR):
        print(f"错误: 输出目录 {OUTPUT_DIR} 不存在。请先运行 HDR/LDR 生成脚本。")
        raise SystemExit(1)

    '''
    scenes = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DIR, f))
    ]

    print(f"发现 {len(scenes)} 个场景任务，开始 Dense Event 渲染...")
    for i, scene_path in enumerate(scenes):
        process_scene_folder(scene_path, obj_map)
        print(f"总体进度: {i+1}/{len(scenes)}")
