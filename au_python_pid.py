#!/usr/bin/env python3
"""
Autonomous Vehicle Controller - Following C Code Pattern
Key differences from previous version:
1. Uses camera FOV for angle calculation (like C code)
2. Simple average filter (no complex state estimation initially)
3. Direct pixel access method from C code
4. PID controller as baseline (matching C implementation)
"""

import os
import math
import numpy as np
from collections import deque

# Webots imports
from vehicle import Driver
from controller import Robot

# ==================== CONFIGURATION (matching C code style) ====================
# PID parameters (from C code)
KP = 0.25
KI = 0.006
KD = 2.0

# Additional control gains
K_BASELINE = 1.0  # Baseline correction
GAIN_RL = 0.3     # RL correction weight

# Speed control
MAX_V = 12.0
DESIRED_SPEED = 50.0  # km/h (C code starts at 50)
MIN_SPEED = 20.0      # km/h

# Steering limits
MAX_STEER = 0.5
MAX_DSTEER = 0.1

# Camera parameters (C code style)
FILTER_SIZE = 3  # Yellow line angle filter size from C code
UNKNOWN = 99999.99  # Sentinel value from C code

# LiDAR parameters
OBSTACLE_THRESHOLD = 20.0  # meters (C code uses 20.0)
CRITICAL_DISTANCE = 10.0

# ==================== UTILITY CLASSES ====================
class AngleFilter:
    """Simple average filter for yellow line angle (from C code)"""
    def __init__(self, size=FILTER_SIZE):
        self.size = size
        self.values = [0.0] * size
        self.first_call = True
    
    def filter_angle(self, new_value):
        """Filter angle of the yellow line (simple average)"""
        if self.first_call or new_value == UNKNOWN:
            # Reset all old values to 0.0
            self.first_call = False
            self.values = [0.0] * self.size
        else:
            # Shift old values
            for i in range(self.size - 1):
                self.values[i] = self.values[i + 1]
        
        if new_value == UNKNOWN:
            return UNKNOWN
        else:
            self.values[self.size - 1] = new_value
            return sum(self.values) / self.size

class PIDController:
    """PID controller for line following (from C code)"""
    def __init__(self, kp=KP, ki=KI, kd=KD):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.old_value = 0.0
        self.integral = 0.0
        self.need_reset = False
    
    def apply_pid(self, yellow_line_angle):
        """Apply PID control (matching C code logic)"""
        if self.need_reset:
            self.old_value = yellow_line_angle
            self.integral = 0.0
            self.need_reset = False
        
        # Anti-windup mechanism (from C code)
        if (yellow_line_angle >= 0) != (self.old_value >= 0):
            self.integral = 0.0
        
        diff = yellow_line_angle - self.old_value
        
        # Limit integral (from C code)
        if -30 < self.integral < 30:
            self.integral += yellow_line_angle
        
        self.old_value = yellow_line_angle
        return self.kp * yellow_line_angle + self.ki * self.integral + self.kd * diff
    
    def reset(self):
        self.need_reset = True

# ==================== WEBOTS SETUP ====================
driver = Driver()
timestep = int(driver.getBasicTimeStep())
dt = timestep / 1000.0

# Initialize devices
gps = driver.getDevice("gps")
gyro = driver.getDevice("gyro")
camera = driver.getDevice("camera")
lidar = driver.getDevice("Sick LMS 291")
display = driver.getDevice("display")

has_gps = gps is not None
has_gyro = gyro is not None
has_camera = camera is not None
has_lidar = lidar is not None
has_display = display is not None

if has_gps:
    gps.enable(timestep)
if has_gyro:
    gyro.enable(timestep)

# Camera setup
camera_width = -1
camera_height = -1
camera_fov = -1.0

if has_camera:
    camera.enable(timestep)
    camera_width = camera.getWidth()
    camera_height = camera.getHeight()
    camera_fov = camera.getFov()
    print(f"✓ Camera: {camera_width}x{camera_height}, FOV={math.degrees(camera_fov):.1f}°")

# LiDAR setup
sick_width = -1
sick_fov = -1.0

if has_lidar:
    lidar.enable(timestep)
    sick_width = lidar.getHorizontalResolution()
    sick_fov = lidar.getFov()
    print(f"✓ LiDAR: {sick_width} points, FOV={math.degrees(sick_fov):.1f}°")

if has_display:
    display_width = display.getWidth()
    display_height = display.getHeight()
    print(f"✓ Display: {display_width}x{display_height}")

print(f"✓ Devices: GPS={has_gps}, Gyro={has_gyro}, Camera={has_camera}, LiDAR={has_lidar}")

# ==================== RL POLICY ====================
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

class PolicyNet(torch.nn.Module if TORCH_AVAILABLE else object):
    def __init__(self):
        if TORCH_AVAILABLE:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 2)
            )
    
    def forward(self, x):
        return self.net(x)

class TrainedPolicy:
    def __init__(self, weights_path="./policy_weights.pth", stats_path="./normalization.npz"):
        self.active = False
        self.obs_mean = np.zeros(3)
        self.obs_std = np.ones(3)
        
        if not TORCH_AVAILABLE:
            return
            
        if os.path.exists(stats_path):
            try:
                data = np.load(stats_path)
                loaded_mean = data.get('mean', None)
                loaded_std = data.get('std', None)
                if loaded_mean is not None and len(loaded_mean) >= 3:
                    self.obs_mean = loaded_mean[:3]
                    self.obs_std = loaded_std[:3] if loaded_std is not None else np.ones(3)
                    print(f"✓ RL normalization loaded")
            except Exception as e:
                print(f"⚠️ Normalization error: {e}")
        
        if os.path.exists(weights_path):
            try:
                self.net = PolicyNet()
                checkpoint = torch.load(weights_path, map_location='cpu')
                
                state_dict = None
                if isinstance(checkpoint, dict):
                    for key in ['policy', 'model_state_dict', 'state_dict', 'actor_state_dict']:
                        if key in checkpoint:
                            state_dict = checkpoint[key]
                            break
                    if state_dict is None:
                        state_dict = {k.replace('actor.', '').replace('policy.', ''): v 
                                     for k, v in checkpoint.items()}
                else:
                    state_dict = checkpoint
                
                self.net.load_state_dict(state_dict, strict=False)
                self.net.eval()
                self.active = True
                print("✓ RL policy loaded")
            except Exception as e:
                print(f"⚠️ RL policy error: {e}")
    
    def predict(self, v_kmh, yellow_line_angle):
        """Returns: steering_correction"""
        if not self.active:
            return 0.0
        
        try:
            # Simple observation: just velocity and line angle
            v_ms = v_kmh / 3.6
            obs = np.array([v_ms, yellow_line_angle, 0.0], dtype=np.float32)
            obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-8)
            obs_norm = np.clip(obs_norm, -10.0, 10.0)
            
            with torch.no_grad():
                x = torch.FloatTensor(obs_norm).unsqueeze(0)
                output = self.net(x).squeeze(0).numpy()
            
            steer_corr = float(np.clip(output[1], -0.3, 0.3))
            return steer_corr
        except Exception:
            return 0.0

policy = TrainedPolicy()

# ==================== CAMERA PROCESSING (C code style) ====================
def color_diff(pixel, ref):
    """Compute RGB difference (from C code)"""
    diff = 0
    for i in range(3):
        d = pixel[i] - ref[i]
        diff += abs(d)
    return diff

def process_camera_image(image, width, height, fov):
    """
    Process camera image - EXACT C code logic
    Returns approximate angle of yellow road line or UNKNOWN
    """
    if image is None:
        return UNKNOWN
    
    num_pixels = height * width
    REF = [95, 187, 203]  # Road yellow (BGR format) - FROM C CODE
    sumx = 0
    pixel_count = 0
    
    # Iterate through all pixels (C code does this)
    for x in range(num_pixels):
        pixel_idx = x * 4  # BGRA format
        try:
            # Extract BGR (skip alpha)
            b = image[pixel_idx] if isinstance(image[pixel_idx], int) else ord(image[pixel_idx])
            g = image[pixel_idx + 1] if isinstance(image[pixel_idx + 1], int) else ord(image[pixel_idx + 1])
            r = image[pixel_idx + 2] if isinstance(image[pixel_idx + 2], int) else ord(image[pixel_idx + 2])
            
            pixel = [b, g, r]
            
            if color_diff(pixel, REF) < 30:  # Threshold from C code
                sumx += x % width  # Get x coordinate
                pixel_count += 1
        except (IndexError, TypeError):
            continue
    
    # If no pixels detected...
    if pixel_count == 0:
        return UNKNOWN
    
    # Calculate angle using camera FOV (EXACT C code formula)
    avg_x = sumx / pixel_count
    normalized = avg_x / width - 0.5  # Range: -0.5 to 0.5
    angle = normalized * fov
    
    return angle

def visualize_camera(cam, disp, step_count):
    """Simple visualization"""
    if not disp or step_count % 10 != 0:
        return
    
    try:
        width = cam.getWidth()
        height = cam.getHeight()
        
        # Clear and draw basics
        disp.setColor(0x000000)
        disp.fillRectangle(0, 0, width, height)
        
        # Draw camera image
        img = cam.getImage()
        if img:
            # Draw center line
            disp.setColor(0x0000FF)
            disp.drawLine(width//2, 0, width//2, height-1)
            
            # Detect and mark yellow pixels
            REF = [95, 187, 203]
            disp.setColor(0x00FF00)
            
            for y in range(0, height, 8):
                for x in range(0, width, 8):
                    idx = (y * width + x) * 4
                    try:
                        b = img[idx] if isinstance(img[idx], int) else ord(img[idx])
                        g = img[idx+1] if isinstance(img[idx+1], int) else ord(img[idx+1])
                        r = img[idx+2] if isinstance(img[idx+2], int) else ord(img[idx+2])
                        
                        if color_diff([b,g,r], REF) < 30:
                            disp.drawPixel(x, y)
                    except:
                        continue
            
            # Text
            disp.setColor(0xFFFFFF)
            disp.setFont("Arial", 10, True)
            disp.drawText(f"Yellow Line Detection", 5, 5)
    except Exception as e:
        pass

# ==================== LIDAR PROCESSING (C code style) ====================
def process_sick_data(sick_data, width, fov):
    """
    Process SICK LiDAR data - C code logic
    Returns: (obstacle_angle, obstacle_dist) or (UNKNOWN, 0.0)
    """
    if sick_data is None or len(sick_data) == 0:
        return UNKNOWN, 0.0
    
    HALF_AREA = 20  # Check 20 degrees wide middle area (from C code)
    sumx = 0
    collision_count = 0
    obstacle_dist = 0.0
    
    center = width // 2
    for x in range(center - HALF_AREA, center + HALF_AREA):
        if x < 0 or x >= len(sick_data):
            continue
        
        range_val = sick_data[x]
        if range_val < 20.0:  # C code threshold
            sumx += x
            collision_count += 1
            obstacle_dist += range_val
    
    # If no obstacle detected...
    if collision_count == 0:
        return UNKNOWN, 0.0
    
    obstacle_dist = obstacle_dist / collision_count
    # Calculate angle (EXACT C code formula)
    avg_x = sumx / collision_count
    normalized = avg_x / width - 0.5
    obstacle_angle = normalized * fov
    
    return obstacle_angle, obstacle_dist

# ==================== CONTROL SYSTEM ====================
angle_filter = AngleFilter(FILTER_SIZE)
pid_controller = PIDController(KP, KI, KD)

def set_speed_smooth(current_speed, target_speed, max_change=5.0):
    """Smooth speed changes (C code increments by 5 km/h)"""
    if target_speed > current_speed:
        return min(current_speed + max_change, target_speed)
    elif target_speed < current_speed:
        return max(current_speed - max_change, target_speed)
    return target_speed

def set_steering_angle(current_angle, target_angle, max_change=0.1):
    """Limit steering angle changes (from C code)"""
    # Limit the difference with previous steering_angle
    if target_angle - current_angle > max_change:
        target_angle = current_angle + max_change
    if target_angle - current_angle < -max_change:
        target_angle = current_angle - max_change
    
    # Limit range of the steering angle
    if target_angle > MAX_STEER:
        target_angle = MAX_STEER
    elif target_angle < -MAX_STEER:
        target_angle = -MAX_STEER
    
    return target_angle

# ==================== MAIN CONTROL LOOP ====================
print("\n" + "="*60)
print("AUTONOMOUS VEHICLE CONTROLLER - C CODE PATTERN")
print("="*60 + "\n")

# Start vehicle (like C code)
current_speed = DESIRED_SPEED
driver.setCruisingSpeed(current_speed)
driver.setHazardFlashers(True)
driver.setDippedBeams(True)

steering_angle = 0.0
step_count = 0
lost_line_count = 0

print("Starting auto-drive mode...")

while driver.step() != -1:
    current_time = driver.getTime()
    
    # ========== READ SENSORS ==========
    camera_image = None
    sick_data = None
    
    if has_camera:
        camera_image = camera.getImage()
    
    if has_lidar:
        sick_data = lidar.getRangeImage()
    
    # ========== PROCESS CAMERA (C code logic) ==========
    yellow_line_angle = UNKNOWN
    if camera_image is not None:
        raw_angle = process_camera_image(camera_image, camera_width, camera_height, camera_fov)
        yellow_line_angle = angle_filter.filter_angle(raw_angle)
    
    # ========== PROCESS LIDAR (C code logic) ==========
    obstacle_angle = UNKNOWN
    obstacle_dist = 0.0
    if sick_data is not None:
        obstacle_angle, obstacle_dist = process_sick_data(sick_data, sick_width, sick_fov)
    
    # ========== CONTROL LOGIC (following C code structure) ==========
    if yellow_line_angle != UNKNOWN or obstacle_angle != UNKNOWN:
        driver.setBrakeIntensity(0.0)
        
        # Obstacle avoidance (C code logic)
        if obstacle_angle != UNKNOWN:
            obstacle_steering = steering_angle
            if 0.0 < obstacle_angle < 0.4:
                obstacle_steering = steering_angle + (obstacle_angle - 0.25) / obstacle_dist
            elif obstacle_angle > -0.4:
                obstacle_steering = steering_angle + (obstacle_angle + 0.25) / obstacle_dist
            
            # Line following + obstacle avoidance
            if yellow_line_angle != UNKNOWN:
                line_following_steering = pid_controller.apply_pid(yellow_line_angle)
                
                # Get RL correction
                rl_correction = policy.predict(current_speed, yellow_line_angle)
                line_following_steering += rl_correction * GAIN_RL
                
                # Choose steering (C code logic)
                if obstacle_steering > 0 and line_following_steering > 0:
                    steer = max(obstacle_steering, line_following_steering)
                elif obstacle_steering < 0 and line_following_steering < 0:
                    steer = min(obstacle_steering, line_following_steering)
                else:
                    steer = line_following_steering
            else:
                steer = obstacle_steering
                pid_controller.reset()
            
            steering_angle = set_steering_angle(steering_angle, steer)
            lost_line_count = 0
            
        elif yellow_line_angle != UNKNOWN:
            # No obstacle - just follow line (C code)
            line_steering = pid_controller.apply_pid(yellow_line_angle)
            
            # Add RL correction
            rl_correction = policy.predict(current_speed, yellow_line_angle)
            line_steering += rl_correction * GAIN_RL
            
            steering_angle = set_steering_angle(steering_angle, line_steering)
            lost_line_count = 0
    else:
        # Lost the line - brake (C code behavior)
        driver.setBrakeIntensity(0.4)
        pid_controller.reset()
        lost_line_count += 1
        
        # Reduce speed when lost
        if lost_line_count > 20:
            target_speed = MIN_SPEED
            current_speed = set_speed_smooth(current_speed, target_speed, 2.0)
            driver.setCruisingSpeed(current_speed)
    
    # Apply steering
    driver.setSteeringAngle(steering_angle)
    
    # Visualize
    if has_display and has_camera:
        visualize_camera(camera, display, step_count)
    
    # ========== LOGGING ==========
    if step_count % 20 == 0:
        line_status = "LINE" if yellow_line_angle != UNKNOWN else f"LOST({lost_line_count})"
        obs_status = f"obs={obstacle_dist:.1f}m" if obstacle_angle != UNKNOWN else "clear"
        
        if yellow_line_angle != UNKNOWN:
            angle_str = f"{math.degrees(yellow_line_angle):+7.2f}"
        else:
            angle_str = "UNKNOWN"
        
        print(f"[{current_time:6.2f}s] {line_status:12s} | "
              f"angle={angle_str}° steer={math.degrees(steering_angle):+.1f}° | "
              f"speed={current_speed:.0f}km/h {obs_status}")
    
    step_count += 1

print("\nController stopped")