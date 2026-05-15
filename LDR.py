import bpy
import os
import json
import math
import mathutils
import random # 仅用于兼容性，实际逻辑不随机

# ================= 配置区域 =================
# 【注意】必须与 HDR 渲染时的路径完全一致，以便找到原文件
OBJ_DIR = r"F:/TreeOBJ/stls"
HDRI_DIR = r"E:/dataSet/myblendevent/hdr_dataset"
OUTPUT_DIR = r"F:/TreeOBJ/reflective_raw" 
LOG_FILE = r"F:/TreeOBJ/skipped_ldr.txt"

# 渲染设置
RESOLUTION_X = 640
RESOLUTION_Y = 480
RENDER_SAMPLES = 64 # LDR 采样数可以比 HDR 低一些，64通常足够
FRAME_COUNT = 120
FPS = 120

# 【核心配置】曝光序列 (n=5)
# EV 0 是基准，+1 是亮一倍，-1 是暗一倍 (2^ev)
# 对应线性乘数: 0.25x, 0.5x, 1.0x, 2.0x, 4.0x
EXPOSURE_LEVELS = [-3,-2, -1, 3,1, 2] 
# EXPOSURE_LEVELS = [-5,-10,10, 5] 
# ================= 辅助函数 =================

def log_error(msg):
    print(f"[Error] {msg}")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")

def get_obj_map(folder):
    """
    建立文件名到绝对路径的映射。
    用于根据 metadata 中的 object_id 快速找到原始模型文件。
    """
    mapping = {}
    valid_exts = ('.obj', '.stl', '.glb', '.gltf', '.ply')
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(valid_exts):
                # 存两种key：带后缀和不带后缀，防止ID记录方式不同导致的查找失败
                full_path = os.path.join(root, f)
                mapping[f] = full_path
                name_no_ext = os.path.splitext(f)[0]
                mapping[name_no_ext] = full_path
    return mapping

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.materials: bpy.data.materials.remove(block)
    for block in bpy.data.images: bpy.data.images.remove(block)
    for block in bpy.data.cameras: bpy.data.cameras.remove(block)

def setup_renderer_ldr():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for device in prefs.devices: device.use = True
    except: pass
    
    # 强制关闭 GPU 细分 (防止崩溃)
    try: bpy.context.preferences.viewport.use_gpu_subdivision = False
    except: pass
    
    # 渲染参数
    scene.cycles.max_bounces = 12
    scene.cycles.glossy_bounces = 8
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    
    # === 【关键设置：LDR 视图变换】 ===
    # 使用 'Standard'。不要用 'AgX' 或 'Filmic'。
    # Standard 会直接截断 >1.0 的数值，这对于生成 Ground Truth 数据集是必须的。
    scene.view_settings.view_transform = 'Standard' 
    scene.view_settings.look = 'None'
    
    # 输出格式 PNG 8-bit
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'

def normalize_object_raw(obj):
    """
    【必须与 HDR 脚本完全一致】
    只归一化尺寸和居中，不修改几何拓扑
    """
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

def reconstruct_material(obj, mat_data):
    """
    根据 metadata.json 中的记录，1:1 还原材质
    """
    mat = bpy.data.materials.new(name=mat_data["name"])
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    
    def safe_set(input_names, value):
        if not isinstance(input_names, list): input_names = [input_names]
        for name in input_names:
            if name in bsdf.inputs:
                bsdf.inputs[name].default_value = value
                return

    # 1. 恢复颜色
    color = mat_data.get("color", (0.8, 0.8, 0.8, 1.0))
    safe_set(["Base Color", "Color"], color)
    
    # 2. 恢复金属度 (根据记录的数据或类型反推)
    is_metallic = mat_data.get("metallic", 0.0) 
    # 如果 metadata 里存的是 type 字符串
    if "type" in mat_data and mat_data["type"] == "metallic":
        is_metallic = 1.0
    safe_set("Metallic", is_metallic)
    
    # 3. 其他参数
    safe_set("Roughness", mat_data.get("roughness", 0.5))
    safe_set(["Coat Weight", "Clearcoat"], mat_data.get("coat_weight", 0.0))
    safe_set(["Coat Roughness", "Clearcoat Roughness"], mat_data.get("coat_roughness", 0.03))
    safe_set(["Anisotropic", "Anisotropic Rotation"], mat_data.get("anisotropic", 0.0))
    
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)

def reconstruct_hdri(hdri_filename):
    hdri_path = os.path.join(HDRI_DIR, hdri_filename)
    if not os.path.exists(hdri_path): 
        print(f"Warning: HDRI not found {hdri_path}")
        return
    
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    
    node_out = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    node_bg = world.node_tree.nodes.new(type='ShaderNodeBackground')
    node_env = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
    
    node_env.image = bpy.data.images.load(hdri_path)
    node_bg.inputs['Strength'].default_value = 1.5 # 保持与 HDR 脚本一致的强度
    
    links = world.node_tree.links
    links.new(node_env.outputs['Color'], node_bg.inputs['Color'])
    links.new(node_bg.outputs['Background'], node_out.inputs['Surface'])

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

# ================= 主流程 =================

def process_scene_folder(scene_folder_path, obj_mapping):
    # 1. 读取 metadata
    json_path = os.path.join(scene_folder_path,"HDR", "metadata.json")
    if not os.path.exists(json_path): return

    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    scene_id = os.path.basename(scene_folder_path)
    
    # 2. 检查是否已完成 (查看最后一档曝光文件夹是否有文件)
    last_ev_folder = os.path.join(scene_folder_path, "LDR", f"ev_{EXPOSURE_LEVELS[-1]}")
    if os.path.exists(last_ev_folder) and len(os.listdir(last_ev_folder)) > 0:
        print(f"Skipping {scene_id} (LDR exists)")
        return

    print(f"--- 生成 LDR 序列: {scene_id} ---")
    clean_scene()
    setup_renderer_ldr()
    
    # 3. 找回原始模型文件
    obj_id = meta["object_id"]
    # 尝试直接匹配 ID，或添加后缀匹配
    obj_path = obj_mapping.get(obj_id)
    if not obj_path:
        log_error(f"Missing Model Source: {obj_id} (in scene {scene_id})")
        return


    # 4. 导入模型 (使用与原脚本一致的导入逻辑)
    try:
        ext = os.path.splitext(obj_path)[1].lower()
        if ext == '.obj':
            if hasattr(bpy.ops.wm, 'obj_import'): bpy.ops.wm.obj_import(filepath=obj_path)
            else: bpy.ops.import_scene.obj(filepath=obj_path)
        elif ext == '.stl':
            if hasattr(bpy.ops.wm, 'stl_import'): bpy.ops.wm.stl_import(filepath=obj_path)
            else: bpy.ops.import_mesh.stl(filepath=obj_path)
        elif ext in ['.glb', '.gltf']: bpy.ops.import_scene.gltf(filepath=obj_path)
        elif ext == '.ply':
            if hasattr(bpy.ops.wm, 'ply_import'): bpy.ops.wm.ply_import(filepath=obj_path)
            else: bpy.ops.import_mesh.ply(filepath=obj_path)
    except Exception as e:
        log_error(f"Import Error {obj_id}: {e}")
        return

    imported_obj = bpy.context.selected_objects[0]
    
    # 5. 应用一致的几何处理 (关键步骤，保证位置对齐)
    normalize_object_raw(imported_obj) 
    
    # 6. 重建环境 (材质、HDRI、相机)
    reconstruct_material(imported_obj, meta["material"])
    reconstruct_hdri(meta["hdri_file"])
    reconstruct_camera_anim(imported_obj)
    
    # 7. 多重曝光渲染循环 (Linear Sequence)
    for ev in EXPOSURE_LEVELS:
        # 设置曝光值
        bpy.context.scene.view_settings.exposure = ev
        
        # 创建子文件夹: e.g., OUTPUT_DIR/SceneID/LDR/ev_-2
        save_path = os.path.join(scene_folder_path, "LDR", f"ev_{ev}")
        os.makedirs(save_path, exist_ok=True)
        
        bpy.context.scene.render.filepath = os.path.join(save_path, "frame_")
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = FRAME_COUNT
        
        print(f"  > Rendering EV {ev} ...")
        try:
            bpy.ops.render.render(animation=True)
        except Exception as e:
            log_error(f"Render Error {scene_id} EV {ev}: {e}")
            break

if __name__ == "__main__":
    # 1. 首先建立模型库的索引，以免找不到文件
    print("正在索引原始模型库 (OBJ_DIR)...")
    obj_map = get_obj_map(OBJ_DIR)
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"错误: 输出目录 {OUTPUT_DIR} 不存在。请先运行生成 HDR 的脚本。")
    else:
        # 2. 遍历输出目录下的所有场景文件夹
        scenes = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, f))]
        print(f"发现 {len(scenes)} 个场景任务，开始 LDR 序列渲染...")
        
        for i, scene_path in enumerate(scenes):
            process_scene_folder(scene_path, obj_map)
            print(f"总体进度: {i+1}/{len(scenes)}")