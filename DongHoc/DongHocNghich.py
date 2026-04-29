import mujoco
import numpy as np
import time

model = mujoco.MjModel.from_xml_path('ur5e.xml')
data = mujoco.MjData(model)
site_name = 'attachment_site'
site_id = model.site(site_name).id

# VỊ TRÍ ĐÍCH VD    
target_pos = np.array([0.4, 0.2, 0.4])

# Tham số điều khiển IK
step_size = 0.5      # Toc do dich chuyen/ hoi tu

for i in range(1000):
    mujoco.mj_forward(model, data)        # Cập nhật động học thuận => vị trí hiện tại
    current_pos = data.site_xpos[site_id]
    error = target_pos - current_pos      # Tính sai lệch vị trí hiện tại và mong muốn
    
    # Kiểm tra về đích
    if np.linalg.norm(error) < 0.001: # Sai số cho phép (1mm)
        print(f"> Đã hội tụ sau {i} bước!")
        break
    
    # Jacobian (3 hàng Position (3x6), 3 hàng Orientation)
    jacp = np.zeros((3, model.nv)) # Jacobian vị trí
    jacr = np.zeros((3, model.nv)) # Jacobian xoay
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    
    # dot x= J_T * dot q => dot q = J_T^-1 * dot x
    j_inv = np.linalg.pinv(jacp) #J_T^-1
    dq = j_inv @ error #dq: số rad cần quay
    
    # Cập nhật góc khớp: q = q + dq (* step_size)
    data.qpos[:6] += dq * step_size

print(f"Khớp cần quay (Radian): {data.qpos[:6]}")
print(f"Vị trí thực tế đạt được: {data.site_xpos[site_id]}")

# Hiện cửa sổ mô phỏng
import mujoco.viewer
data.ctrl[:6] = data.qpos[:6]
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()        

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

