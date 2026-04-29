import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("scene+grasp.xml")
data = mujoco.MjData(model)

# ID
site_id = model.site('attachment_site').id
box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")

wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link") 
eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "magnet_constraint")

# Tham số điều khiển
Kp = np.array([2000, 3000, 2000, 800, 300, 100])
Kd = np.array([200, 300, 200, 80, 30, 10])
dt = model.opt.timestep

def move_to_target(target_pos, duration=3.0, viewer=None, q_target_posture=None):
    start_time = data.time
    target_quat = np.array([0, 0.7071, 0.7071, 0]) 
    if q_target_posture is None:
        q_target_posture = np.array([0, -1.2, 1.5, -1.8, -1.57, 0])

    while data.time - start_time < duration:
        step_start = time.time()
        error_pos = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(error_pos) < 0.005:
            print("Đã tới target")
            break

        error_ori = np.zeros(3)
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id])
        res_quat = np.zeros(4)
        inv_site_quat = np.zeros(4)

        mujoco.mju_negQuat(inv_site_quat, site_quat)
        mujoco.mju_mulQuat(res_quat, target_quat, inv_site_quat)
        mujoco.mju_quat2Vel(error_ori, res_quat, 1.0)

        error_full = np.concatenate([error_pos, error_ori])

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac_full = np.vstack([jacp[:, :6], jacr[:, :6]])
        jac_inv = jac_full.T @ np.linalg.inv( jac_full @ jac_full.T + 1e-4 * np.eye(6) )

        dq_main = jac_inv @ (5.0 * error_full)

        null_space_filter = np.eye(6) - (jac_inv @ jac_full)
        dq_null = null_space_filter @ (    5.0 * (q_target_posture - data.qpos[:6])  )

        dq_ideal = dq_main + dq_null
        data.ctrl[:6] = (       Kp * (dq_ideal * dt)  + Kd * (dq_ideal - data.qvel[:6])    + data.qfrc_bias[:6] )

        mujoco.mj_step(model, data)

        if viewer:
            viewer.sync()

        elapsed = time.time() - step_start

        if elapsed < dt:
            time.sleep(dt - elapsed)

# Control
with mujoco.viewer.launch_passive(model, data) as viewer:
    data.qpos[:6] = [0, -1.2, 1.5, -1.8, -1.57, 0]
    mujoco.mj_forward(model, data)
    time.sleep(1)

    # BƯỚC 1: Di chuyển tới điểm gap
    temp_box_pos = data.xpos[box_id].copy()
    pre_grasp = temp_box_pos + np.array([0, 0, 0.3])
    print("Tới điểm gap...")
    move_to_target(pre_grasp, duration=2.0, viewer=viewer)

    # BƯỚC 2: TÍNH CHIỀU CAO Z của cái hộp 
    real_box_pos = data.xpos[box_id].copy()
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
    h_height = model.geom_size[geom_id][2]
    
    grasp_point = np.array([real_box_pos[0], real_box_pos[1], real_box_pos[2] + h_height])
    print("Hạ xuống gắp...")
    move_to_target(grasp_point, duration=1.5, viewer=viewer)

    # BƯỚC 3: Kích hoạt nam châm
    print("Đang gắp...")


    print("\n>> GẮP")

    # Lấy tư thế khớp hiện tại robot 
    q_current = data.qpos[:6].copy()

    for _ in range(50):
        error_q = q_current - data.qpos[:6]
        data.ctrl[:6] = Kp * error_q - Kd * data.qvel[:6] + data.qfrc_bias[:6]
        mujoco.mj_step(model, data)

    # 2.  vị trí/hướng của Site trên Robot
    site_robot_pos = data.site_xpos[site_id].copy()
    site_robot_mat = data.site_xmat[site_id].reshape(3, 3).copy()
    
    # 3. Tính toán vtri Body hop
    box_attach_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "box_attach_site")
    site_box_local_pos = model.site_pos[box_attach_site_id]
    
    target_quat = np.zeros(4)
    mujoco.mju_mat2Quat(target_quat, site_robot_mat.flatten())

    # Site body trùng với Site Robot
    offset_world = site_robot_mat @ site_box_local_pos
    box_body_target_pos = site_robot_pos - offset_world

    # 4. THỰC HIỆN TELEPORT
    box_qpos_adr = model.jnt_qposadr[model.body_jntadr[box_id]]
    data.qpos[box_qpos_adr : box_qpos_adr+3] = box_body_target_pos
    data.qpos[box_qpos_adr+3 : box_qpos_adr+7] = target_quat

    # 5. BẬT NAM CHÂM
    mujoco.mj_forward(model, data) 
    data.eq_active[eq_id] = 1
    print("Bat nam cham.")

    # 6. DUY TRÌ TRẠNG THÁI 
    for _ in range(150):
        error_q = q_current - data.qpos[:6]
        data.ctrl[:6] = Kp * error_q - Kd * data.qvel[:6] + data.qfrc_bias[:6]
        
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
        time.sleep(dt)

    print("Di chuyển.")


    # 1. Nhấc hộp lên c
    lift_target = site_robot_pos + np.array([0, 0, 0.2]) 
    print("Đang nhấc vật lên...")
    move_to_target(lift_target, duration=0.5, viewer=viewer)

    # 2. Di chuyển đen đích 
    delivery_overhead = np.array([-0.5, -0.4, 0.3]) 
    print("Di chuyển đen đích...")
    move_to_target(delivery_overhead, duration=0.5, viewer=viewer)

    # 3. HẠ XUỐNG 
    delivery_touchdown = np.array([-0.5, -0.4, 0.10]) 
    print("Hạ vật...")
    move_to_target(delivery_touchdown, duration=0.5, viewer=viewer)

    # 4. THẢ VẬT
    print("Tắt nam châm.")
    data.eq_active[eq_id] = 0 # Tắt kết nối vật lý

    # 5. NHẤC LÊN 
    final_retreat = delivery_touchdown + np.array([0, 0, 0.15])
    print("Nhấc cánh tay lên...")
    move_to_target(final_retreat, duration=1.5, viewer=viewer)

    while viewer.is_running():
        step_start = time.time()
        #  duy trì bộ điều khiển PD để chống lại trọng lượng cái hộp + giữ robot đứng yên tại vị trí gắp (q_current)
        error_q = q_current - data.qpos[:6]
        data.ctrl[:6] = Kp * error_q - Kd * data.qvel[:6] + data.qfrc_bias[:6]

        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)