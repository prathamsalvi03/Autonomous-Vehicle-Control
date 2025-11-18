#!/usr/bin/env python3
"""
RL + MPC Autonomous Vehicle Controller
(70% Camera + 30% RL reference blending)
"""

import os
import math
import numpy as np
import traceback
from time import time
from vehicle import Driver

# ==================== CONFIG ====================
N = 10
MAX_STEER = 0.5
MAX_DSTEER = 0.15
MAX_ACCEL = 2.0
MAX_VEL = 15.0
Q_LATERAL, Q_VEL = 200.0, 30.0
R_STEER, R_ACCEL, R_DSTEER = 1.0, 0.5, 10.0
REG_X = 1e-3
L = 2.5
DESIRED_SPEED_KMH, MIN_SPEED_KMH, MAX_SPEED_KMH = 5.0, 5.0, 20.0
SPEED_CAP_FOR_MPC_KMH = 25.0
FILTER_SIZE = 5
UNKNOWN = 99999.99

def clamp(v, lo, hi): return max(lo, min(hi, v))

class AngleFilter:
    def __init__(self, size=FILTER_SIZE):
        self.size = size; self.values = [0.0]*size; self.first_call = True
    def filter_angle(self, new_value):
        if self.first_call:
            self.first_call = False
            if new_value != UNKNOWN:
                self.values = [new_value]*self.size
                return new_value
            return UNKNOWN
        if new_value == UNKNOWN: return UNKNOWN
        self.values = self.values[1:] + [new_value]
        return sum(self.values)/self.size

# ==================== WEBOTS SETUP ====================
driver = Driver()
timestep = int(driver.getBasicTimeStep())
dt = timestep / 1000.0

gps = driver.getDevice("gps"); camera = driver.getDevice("camera")
lidar = driver.getDevice("Sick LMS 291"); gyro = driver.getDevice("gyro")
for d in [gps, camera, lidar, gyro]:
    if d: d.enable(timestep)

if camera:
    camera_width, camera_height, camera_fov = camera.getWidth(), camera.getHeight(), camera.getFov()
else:
    camera_width = camera_height = camera_fov = -1
if lidar:
    sick_width, sick_fov = lidar.getHorizontalResolution(), lidar.getFov()
else:
    sick_width = sick_fov = -1

print("Devices ready.")

# ==================== RL POLICY ====================
TORCH_AVAILABLE = False
try:
    import torch; TORCH_AVAILABLE = True
except Exception: pass

class PolicyNet(torch.nn.Module if TORCH_AVAILABLE else object):
    def __init__(self):
        if TORCH_AVAILABLE:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3,64), torch.nn.ReLU(),
                torch.nn.Linear(64,64), torch.nn.ReLU(),
                torch.nn.Linear(64,2))
    def forward(self, x): return self.net(x)

class HighLevelPolicy:
    def __init__(self):
        self.active = False; self.obs_mean = np.zeros(3); self.obs_std = np.ones(3)
        if not TORCH_AVAILABLE: return
        weights_path = "./policy_weights.pth"; stats_path = "./normalization.npz"
        if os.path.exists(stats_path):
            try:
                data = np.load(stats_path)
                self.obs_mean = data.get('mean', self.obs_mean)[:3]
                self.obs_std = data.get('std', self.obs_std)[:3]
            except Exception: pass
        if os.path.exists(weights_path):
            try:
                self.net = PolicyNet()
                ckpt = torch.load(weights_path, map_location='cpu')
                self.net.load_state_dict(ckpt if not isinstance(ckpt, dict) else ckpt.get('state_dict', ckpt), strict=False)
                self.net.eval(); self.active = True
                print("RL policy loaded.")
            except Exception as e:
                print(f"RL load error: {e}")

    def generate_reference(self, v, lateral_error, dist):
        if not self.active:
            vel = DESIRED_SPEED_KMH/3.6
            if lateral_error != UNKNOWN and abs(lateral_error) > 0.2:
                vel *= (1.0 - 0.5*abs(lateral_error))
            if dist < 20.0:
                vel = min(vel, max(MIN_SPEED_KMH/3.6, dist*0.6))
            return np.clip(vel, MIN_SPEED_KMH/3.6, MAX_SPEED_KMH/3.6), 0.0
        try:
            obs = np.array([v, lateral_error if lateral_error != UNKNOWN else 0.0, dist], dtype=np.float32)
            obs = (obs-self.obs_mean)/(self.obs_std+1e-8)
            with torch.no_grad():
                out = self.net(torch.FloatTensor(obs).unsqueeze(0)).squeeze(0).numpy()
            return float(np.clip(out[0], MIN_SPEED_KMH/3.6, MAX_SPEED_KMH/3.6)), float(np.clip(out[1], -0.3, 0.3))
        except Exception:
            return DESIRED_SPEED_KMH/3.6, 0.0

high_level_policy = HighLevelPolicy()

# ==================== CAMERA PROCESSING ====================
def process_camera_image(image, w, h, fov):
    if image is None or w <= 0 or h <= 0: return UNKNOWN, {}
    try:
        arr = np.frombuffer(image, dtype=np.uint8)
        if arr.size == w*h*4: arr = arr.reshape((h, w, 4))[:, :, :3]
        elif arr.size == w*h*3: arr = arr.reshape((h, w, 3))
        else: return UNKNOWN, {}
        profiles = [{'ref': np.array([95,187,203]), 'th': 30}]
        best = {'pixel_count': 0, 'sumx': 0}
        for p in profiles:
            ref = p['ref'].reshape((1,1,3)); th = p['th']
            diff = np.abs(arr.astype(np.int16) - ref.astype(np.int16))
            mask = (diff.sum(axis=2) < th)
            ys, xs = np.nonzero(mask)
            if xs.size > best['pixel_count']:
                best = {'pixel_count': xs.size, 'sumx': xs.sum()}
        if best['pixel_count'] == 0: return UNKNOWN, best
        avg_x = best['sumx'] / best['pixel_count']
        angle = (avg_x / w - 0.5) * fov
        best['angle'] = angle
        return angle, best
    except Exception:
        return UNKNOWN, {}

# ==================== MPC SETUP ====================
import cvxpy as cp
x_mpc = cp.Variable((2, N+1)); u_mpc = cp.Variable((2, N))
x0_param = cp.Parameter(2); x_ref_param = cp.Parameter((2, N+1)); u_prev_param = cp.Parameter(2)
cost = 0; constraints = [x_mpc[:, 0] == x0_param]
for k in range(N):
    v_next = x_mpc[0,k] + u_mpc[0,k]*dt
    lat_next = x_mpc[1,k] - x_ref_param[0,k]*u_mpc[1,k]*dt
    constraints += [x_mpc[0,k+1]==v_next, x_mpc[1,k+1]==lat_next,
                    u_mpc[0,k]<=MAX_ACCEL, u_mpc[0,k]>=-MAX_ACCEL,
                    u_mpc[1,k]<=MAX_STEER, u_mpc[1,k]>=-MAX_STEER]
    cost += Q_VEL*cp.square(x_mpc[0,k]-x_ref_param[0,k]) + Q_LATERAL*cp.square(x_mpc[1,k]-x_ref_param[1,k])
    cost += R_ACCEL*cp.square(u_mpc[0,k]) + R_STEER*cp.square(u_mpc[1,k])
    if k>0: cost += R_DSTEER*cp.square(u_mpc[1,k]-u_mpc[1,k-1])
mpc_problem = cp.Problem(cp.Minimize(cost), constraints)
print("MPC ready.")

class LowLevelMPC:
    def __init__(self):
        self.prev_control = np.zeros(2)
    def compute_control(self, v, lat, v_ref, lat_ref):
        try:
            x0_param.value = np.array([v, lat])
            x_ref = np.zeros((2,N+1)); x_ref[0,:] = v_ref; x_ref[1,:] = lat_ref
            x_ref_param.value = x_ref; u_prev_param.value = self.prev_control
            mpc_problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if mpc_problem.status not in ["optimal","optimal_inaccurate"]:
                return 0.0, 0.0, False
            accel, steer = float(u_mpc.value[0,0]), float(u_mpc.value[1,0])
            self.prev_control = np.array([accel, steer])
            return accel, steer, True
        except Exception:
            return 0.0, 0.0, False

low_level_mpc = LowLevelMPC()

# ==================== MAIN LOOP ====================
angle_filter = AngleFilter()
driver.setCruisingSpeed(DESIRED_SPEED_KMH)
print("\nController running (70% camera, 30% RL)...\n")

current_vel_ms, prev_pos, step = 0.0, None, 0
while driver.step() != -1:
    gps_pos = gps.getValues()
    if prev_pos is not None:
        dx, dz = gps_pos[0]-prev_pos[0], gps_pos[2]-prev_pos[2]
        current_vel_ms = np.clip(math.hypot(dx, dz)/dt, 0.0, MAX_VEL)
    prev_pos = gps_pos

    img = camera.getImage() if camera else None
    cam_angle, cam_debug = process_camera_image(img, camera_width, camera_height, camera_fov)
    lateral_error = angle_filter.filter_angle(cam_angle)
    if lateral_error == UNKNOWN: lateral_error = 0.0

    # === Blending RL + Camera (30% RL, 70% Camera) ===
    v_rl, lat_rl = high_level_policy.generate_reference(current_vel_ms, lateral_error, 100.0)
    v_cam = DESIRED_SPEED_KMH/3.6; lat_cam = lateral_error
    alpha_cam, alpha_rl = 0.95, 0.05
    v_ref = alpha_cam*v_cam + alpha_rl*v_rl
    lat_ref = alpha_cam*lat_cam + alpha_rl*lat_rl

    accel, steer, ok = low_level_mpc.compute_control(current_vel_ms, lateral_error, v_ref, lat_ref)
    target_kmh = np.clip((current_vel_ms + accel*dt)*3.6, MIN_SPEED_KMH, MAX_SPEED_KMH)
    driver.setSteeringAngle(steer); driver.setCruisingSpeed(target_kmh)

    if step % 20 == 0:
        print(f"[t={driver.getTime():5.1f}s] v={current_vel_ms:4.1f}m/s → {target_kmh:4.0f}km/h | steer={math.degrees(steer):+5.2f}° | lat={math.degrees(lateral_error):+6.2f}°")
    step += 1
