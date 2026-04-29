import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path('ur5e.xml')
data = mujoco.MjData(model)

# PD
Kp = np.array([1500, 1500, 1000, 500, 100, 100]) # bộ tỷ lệ
Kd = np.array([150, 150, 100, 50, 10, 10]) # bộ vi phân

# Vị trí đích (Target joint angles - Radian)
q_target = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0]) 

def pd_control(model, data):
    # Lấy trạng thái hiện tại
    q_current = data.qpos[:6]
    v_current = data.qvel[:6]
    
    # Tính toán sai số
    error = q_target - q_current
    error_dot = 0 - v_current # Vận tốc cuói = 0
    
    F = Kp * error + Kd * error_dot     # Công thức PD: tau = Kp * error + Kd * error_dot
    M_gravity = M_gravity = data.qfrc_bias[:6] #Trọng lực / quán tính/ coriollis
    data.ctrl[:6] = F + M_gravity

# Hiện cửa sổ mô phỏng
site_id = model.site("attachment_site").id
# Tính tọa độ đích XYZ từ q_target
q_init = np.array([0, -1.2, 1.5, -1.8, -1.57, 0])
data.qpos[:6] = q_init
data.qvel[:6] = 0
mujoco.mj_forward(model, data)
target_pos = data.site_xpos[site_id].copy()
l_time = 0 
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        pd_control(model, data)
        mujoco.mj_step(model, data)

        # Chỉ in mỗi 3 giây 
        if time.time() - l_time >= 3:
            current_pos = data.site_xpos[site_id]

            print("Target XYZ :", target_pos)
            print("Current XYZ:", current_pos)

            l_time = time.time()

        viewer.sync()
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)