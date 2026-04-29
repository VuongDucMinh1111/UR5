import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from VaCham import RobotSafety # Import từ file vừa tạo

model = mujoco.MjModel.from_xml_path('ur5e.xml')
data = mujoco.MjData(model)
safety = RobotSafety(model, data)
site_id = model.site('attachment_site').id
x_real = []
y_real = []
z_real = []
# 1. Phương trình quỹ đạo theo t
def trajectory(t):
    return np.array([ 0.4 + 0.1*np.cos(t), 0.1*np.sin(t), 0.3 + 0.02*t ])     # VD: Hình xoắn ốc quanh tâm (0.4, 0, 0.4)

def get_trajectory_pos(t):
    dt = 1e-3
    x = trajectory(t)
    x_next = trajectory(t + dt)
    v = (x_next - x) / dt
    return x, v


# 2. Tham số điều khiển
Kp = np.array([2000, 3000, 2000, 800, 300, 100])
Kd = np.array([200, 300, 200, 80, 30, 10])

#THong số phụ
MAX_REACH = 0.85  # Tầm với max 
t_math = 0.0      # Time ban đầu
TIME_SCALE = 3.0  # Tốc độ mô phỏng
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Reset về vị trí thuận lợi
    data.qpos[:6] = [0, 1.57, 1.57, -0, 0, 0]
    mujoco.mj_forward(model, data)
    start_time = time.time()
    dt = model.opt.timestep
    while viewer.is_running():
        step_start = time.time()
        
        # KIỂM TRA VÙNG LÀM VIỆC TRƯỚC KHI DI CHUYỂN
        next_t_math = t_math + (dt * TIME_SCALE)
        next_x_target, _ = get_trajectory_pos(next_t_math)
       
        distance_from_base = np.linalg.norm(next_x_target) #Tính khoảng cách từ tâm robot (0,0,0) đến điểm đích
        if distance_from_base <= MAX_REACH:
            # Nếu an toàn, cập nhật thời gian t_math và lấy thông số quỹ đạo
            t_math = next_t_math
            x_target, v_target = get_trajectory_pos(t_math)
        else:
            # Nếu ra ngoài tầm với:  # Giữ nguyên vị trí hiện hành của quỹ đạo, và ép vận tốc bằng 0 (t_math = 0)
            x_target, _ = get_trajectory_pos(t_math)
            v_target = np.zeros(3)
        
        # JACOBIAN 
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id) 
        # Sai số vị trí X
        error_pos = x_target - data.site_xpos[site_id]      
        # Tính vận tốc khớp cần thiết V
        dq_ideal = np.linalg.pinv(jacp) @ (v_target + 5.0 * error_pos)
        
        # ĐIỀU KHIỂN PD 
        F = Kp * (dq_ideal * dt) + Kd * (dq_ideal - data.qvel[:6])
        M_gravity = data.qfrc_bias[:6]  
        
        data.ctrl[:6] = F + M_gravity

        if int(data.time) % 3 == 0 and abs(data.time - round(data.time)) < dt:
            current_pos = data.site_xpos[site_id]
            print(f"Target XYZ : {x_target} (Distance: {distance_from_base:.2f}m)")
            print(f"Current XYZ: {current_pos}")
            if distance_from_base > MAX_REACH:
                print(">> Ngoai tầm với! Robot đứng yên.")
        
        mujoco.mj_step(model, data)
        current_pos = data.site_xpos[site_id]

        x_real.append(current_pos[0])
        y_real.append(current_pos[1])
        z_real.append(current_pos[2])
        viewer.sync()
        elapsed = time.time() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_real, y_real, z_real, linewidth=2)

# Điểm bắt đầu
ax.scatter( x_real[0], y_real[0],  z_real[0], s=100, label="Start Position")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Actual End-Effector Trajectory")

ax.legend()
plt.show()