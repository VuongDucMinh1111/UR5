import mujoco
import numpy as np

# 'scene.xml' = môi trường, 'ur5e.xml' = mỗi cánh tay.
model = mujoco.MjModel.from_xml_path('ur5e.xml')
data = mujoco.MjData(model)

# 6 khớp (Radian)
joint_angles = [0, -1.57, 1.57, 0, 0, 0] 
data.qpos[:6] = joint_angles
mujoco.mj_forward(model, data) #Tinh đọng học thuận 

# Vị trí điểm cuối. (Điểm cuối site tên: 'attachment_site' )
site_name = 'attachment_site'
site_id = model.site(site_name).id

pos = data.site_xpos[site_id]     # Tọa độ (x, y, z)
mat = data.site_xmat[site_id]     # Ma trận cosin chi hướng 3x3

print(f">> Kết quả Động học thuận")
print(f"Góc khớp: {joint_angles}")
print(f"Vị trí cuối (x, y, z): {pos}")
print(f"Ma trận xoay:\n{mat.reshape(3,3)}")
T = np.block([[mat.reshape(3, 3), pos.reshape(3, 1)], [0, 0, 0, 1]]); print(T)

# Hiện cửa sổ mô phỏng
import mujoco.viewer
#mujoco.viewer.launch(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        pass