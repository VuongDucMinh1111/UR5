import mujoco
import numpy as np

class RobotSafety:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        # ID các cảm biến 
        self.force_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'force_ee')
        self.torque_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'torque_ee')

    def check_all(self):
        collision = self._check_collisions()
        overforce = self._check_force(threshold=50.0)
        
        if collision or overforce:
            return False # Không an toàn
        return True # OK

    def _check_force(self, threshold):
        if self.force_sid != -1:
            f_adr = self.model.sensor_adr[self.force_sid]
            force = self.data.sensordata[f_adr : f_adr+3]
            if np.linalg.norm(force) > threshold:
                print(f"Cảnh báo lực: {np.linalg.norm(force):.2f}N")
                return True
        return False

    def _check_collisions(self):
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            nm1 = self.model.geom(con.geom1).name
            nm2 = self.model.geom(con.geom2).name
            if "link" in nm1 and "link" in nm2:
                print(f"Tự va chạm: {nm1} - {nm2}")
                return True
        return False