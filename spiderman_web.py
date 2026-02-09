import cv2
import mediapipe as mp
import numpy as np
import math
import random
import pygame
import os

"""sound"""
#Initialize Pygame for sound
pygame.mixer.init()

# load sound effect
try:
    web_sound=pygame.mixer.Sound('web_shoot.mp3')
    sound_available=True
    print("sound effects loaded!")
except:
    print("Sound file not found. Continuing without sound...")
    print("Place 'web_shoot.mp3' in the same folder to enable sound.")
    sound_available = False
        
#Intialize Mediapipe hands
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

#Intialize webcam using opencv
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


"""Gesture Detection"""
def is_spiderman_gesture(hand_landmarks):
    """
    Detect Spider-Man web-shooting gesture:
    - Index finger extended
    - Pinky finger extended
    - Middle and ring fingers folded
    - Thumb can be in any position 
    """
    
    #get finger tip and base positions
    index_tip=hand_landmarks.landmark[8].y
    index_pip=hand_landmarks.landmark[6].y
    
    middle_tip=hand_landmarks.landmark[12].y
    middle_pip=hand_landmarks.landmark[10].y
    
    ring_tip=hand_landmarks.landmark[16].y
    ring_pip=hand_landmarks.landmark[14].y
    
    pinky_pip=hand_landmarks.landmark[20].y
    pinky_tip=hand_landmarks.landmark[18].y
    
    #check if index and pinky are extended (tip is above the pip joint)
    index_extended=index_tip < index_pip - 0.02
    pinky_extended=pinky_tip < pinky_pip - 0.02
    
    #check if middle and ring are folded (tip is below pip joint)
    middle_folded=middle_tip > middle_pip
    ring_folded=ring_tip > ring_pip
    
    return index_extended and pinky_extended and middle_folded and ring_folded

"""SpiderWebeffect"""
class SpiderWebEffect():
    """Creates a radial spider web pattern like the reference images"""
    def __init__(self,center_x,center_y,target_x,target_y,frame_width,frame_height):
        self.center_x = center_x
        self.center_y = center_y
        self.target_x = target_x
        self.target_y = target_y
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Web properties
        self.max_radius = 200
        self.current_radius = 0
        self.growth_speed = 15
        self.num_radial_lines = 12  # Number of spokes
        self.num_rings = 8  # Number of circular rings
        self.active = True
        self.alpha = 1.0
        self.lifetime = 100  # Frames the web stays visible
        self.age = 0
        
        # Calculate angle to target
        dx = target_x - center_x
        dy = target_y - center_y
        self.base_angle = math.atan2(dy, dx)
        
        # Particles for sparkle effect
        self.particles = []
        
        # Web animation
        self.ring_progress = []
        for i in range(self.num_rings):
            self.ring_progress.append(0)
        
        
    def update(self):
        self.age += 1
        
        # Growth phase
        if self.current_radius < self.max_radius:
            self.current_radius += self.growth_speed
            
            # Update ring progress
            for i in range(self.num_rings):
                ring_radius = (i + 1) * (self.max_radius / self.num_rings)
                if self.current_radius > ring_radius:
                    self.ring_progress[i] = min(1.0, self.ring_progress[i] + 0.15)
            
            # Add sparkle particles during growth
            if random.random() < 0.3:
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, self.current_radius)
                px = int(self.center_x + distance * math.cos(angle))
                py = int(self.center_y + distance * math.sin(angle))
                self.particles.append({
                    'x': px,
                    'y': py,
                    'vx': random.uniform(-2, 2),
                    'vy': random.uniform(-2, 2),
                    'life': random.randint(15, 30),
                    'max_life': 30,
                    'size': random.randint(2, 4)
                })
        
        # Fade out phase
        if self.age > self.lifetime:
            self.alpha -= 0.03
            if self.alpha <= 0:
                self.active = False
        
        # Update particles
        for particle in self.particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vx'] *= 0.95
            particle['vy'] *= 0.95
            particle['life'] -= 1
        
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def draw(self,frame):
        if self.alpha <= 0:
            return
        
        overlay = frame.copy()
        
        # Draw radial lines (spokes)
        for i in range(self.num_radial_lines):
            angle = (2 * math.pi * i / self.num_radial_lines) + self.base_angle
            
            # Main radial line
            end_x = int(self.center_x + self.current_radius * math.cos(angle))
            end_y = int(self.center_y + self.current_radius * math.sin(angle))
            
            # Draw with glow effect
            cv2.line(overlay, (self.center_x, self.center_y), (end_x, end_y),
                    (50, 50, 50), 5)  # Outer glow
            cv2.line(overlay, (self.center_x, self.center_y), (end_x, end_y),
                    (0, 0, 0), 3)  # Core line
            cv2.line(overlay, (self.center_x, self.center_y), (end_x, end_y),
                    (0, 0, 0), 1)  # Bright core
        
        # Draw concentric rings
        for ring_idx in range(self.num_rings):
            ring_radius = int((ring_idx + 1) * (self.max_radius / self.num_rings))
            
            if ring_radius <= self.current_radius:
                progress = self.ring_progress[ring_idx]
                
                # Draw curved segments between radial lines
                for i in range(self.num_radial_lines):
                    angle1 = (2 * math.pi * i / self.num_radial_lines) + self.base_angle
                    angle2 = (2 * math.pi * (i + 1) / self.num_radial_lines) + self.base_angle
                    
                    # Create curved web segments
                    points = []
                    num_curve_points = 15
                    for j in range(num_curve_points + 1):
                        t = j / num_curve_points
                        angle = angle1 + (angle2 - angle1) * t
                        
                        # Add slight curvature
                        curve_factor = 1.0 + 0.1 * math.sin(math.pi * t)
                        radius = ring_radius * curve_factor * progress
                        
                        x = int(self.center_x + radius * math.cos(angle))
                        y = int(self.center_y + radius * math.sin(angle))
                        points.append((x, y))
                    
                    # Draw curved segment
                    points_array = np.array(points, np.int32)
                    points_array = points_array.reshape((-1, 1, 2))
                    
                    # Glow effect
                    cv2.polylines(overlay, [points_array], False, (50, 50, 50), 4)
                    # Main line
                    cv2.polylines(overlay, [points_array], False, (0, 0, 0), 2)
        
        # Draw center point (web origin)
        cv2.circle(overlay, (self.center_x, self.center_y), 8, (50, 50, 50), -1)
        cv2.circle(overlay, (self.center_x, self.center_y), 6, (0,0,0), -1)
        cv2.circle(overlay, (self.center_x, self.center_y), 3, (0, 0, 0), -1)
        
        # Draw intersection points on rings
        for ring_idx in range(self.num_rings):
            ring_radius = int((ring_idx + 1) * (self.max_radius / self.num_rings))
            if ring_radius <= self.current_radius:
                for i in range(self.num_radial_lines):
                    angle = (2 * math.pi * i / self.num_radial_lines) + self.base_angle
                    x = int(self.center_x + ring_radius * math.cos(angle))
                    y = int(self.center_y + ring_radius * math.sin(angle))
                    
                    # Draw small connection nodes
                    cv2.circle(overlay, (x, y), 3, (30,30,30), -1)
                    cv2.circle(overlay, (x, y), 2, (0, 0, 0), -1)
        
        # Draw particles (sparkles)
        for particle in self.particles:
            life_ratio = particle['life'] / particle['max_life']
            size = int(particle['size'] * life_ratio)
            if size > 0:
                intensity = int(255 * life_ratio * self.alpha)
                color = (intensity, intensity, min(255, intensity + 80))
                cv2.circle(overlay, (int(particle['x']), int(particle['y'])),
                          size, color, -1)
                # Add glow
                cv2.circle(overlay, (int(particle['x']), int(particle['y'])),
                          size + 2, (intensity // 2, intensity // 2, intensity // 2), 1)
        
        # Blend with alpha
        cv2.addWeighted(overlay, self.alpha * 0.8, frame, 1 - self.alpha * 0.4, 0, frame)

"""ShootingWebStrand"""
class ShootingWebStrand:
    """Creates the shooting web strand that connects to where the web appears"""
    
    def __init__(self, start_x, start_y, end_x, end_y):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.progress = 0
        self.active = True
        self.alpha = 1.0
    
    def update(self):
        if self.progress < 1.0:
            self.progress += 0.15
        else:
            self.alpha -= 0.05
            if self.alpha <= 0:
                self.active = False
    
    def draw(self, frame):
        if self.alpha <= 0:
            return
        
        overlay = frame.copy()
        
        # Current end point based on progress
        current_x = int(self.start_x + (self.end_x - self.start_x) * self.progress)
        current_y = int(self.start_y + (self.end_y - self.start_y) * self.progress)
        
        # Draw shooting strand
        cv2.line(overlay, (self.start_x, self.start_y), (current_x, current_y),
                (180, 180, 255), 6)
        cv2.line(overlay, (self.start_x, self.start_y), (current_x, current_y),
                (255, 255, 255), 3)
        
        # Blend
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha * 0.3, 0, frame)


"""Main application loop"""
#Store active web effects
web_effects = []
shooting_strands = []
gesture_cooldown = 0
last_gesture_state = False

print("=" * 60)
print("🕷️  SPIDER-MAN WEB SHOOTER - RADIAL WEB MODE 🕷️")
print("=" * 60)
print("\nInstructions:")
print("1. Make Spider-Man gesture: Index & Pinky UP, Middle & Ring DOWN")
print("2. Point where you want the web to appear")
print("3. A radial spider web will form at that location!")
print("4. Press 'q' to quit")
print("5. Press 's' to toggle sound")
print("\nStarting camera...")
print("=" * 60)

sound_enabled = True

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    current_gesture = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
            )
            
            if is_spiderman_gesture(hand_landmarks):
                current_gesture = True
                
                if not last_gesture_state and gesture_cooldown == 0:
                    # Get shooting point (index finger tip)
                    index_tip = hand_landmarks.landmark[8]
                    shoot_x = int(index_tip.x * frame_width)
                    shoot_y = int(index_tip.y * frame_height)
                    
                    # Get wrist position
                    wrist = hand_landmarks.landmark[0]
                    wrist_x = int(wrist.x * frame_width)
                    wrist_y = int(wrist.y * frame_height)
                    
                    # Calculate target point (where web will form)
                    # Project forward from hand
                    dx = shoot_x - wrist_x
                    dy = shoot_y - wrist_y
                    distance = 300
                    norm = math.sqrt(dx*dx + dy*dy)
                    if norm > 0:
                        target_x = shoot_x + int((dx / norm) * distance)
                        target_y = shoot_y + int((dy / norm) * distance)
                    else:
                        target_x = shoot_x + distance
                        target_y = shoot_y
                    
                    # Keep target in frame
                    target_x = max(50, min(frame_width - 50, target_x))
                    target_y = max(50, min(frame_height - 50, target_y))
                    
                    # Create shooting strand
                    shooting_strands.append(ShootingWebStrand(
                        shoot_x, shoot_y, target_x, target_y
                    ))
                    
                    # Create radial web at target
                    web_effects.append(SpiderWebEffect(
                        target_x, target_y, shoot_x, shoot_y,
                        frame_width, frame_height
                    ))
                    
                    # Play sound
                    if sound_available and sound_enabled:
                        web_sound.play()
                    
                    gesture_cooldown = 25
                    
                    # Visual feedback
                    cv2.circle(frame, (shoot_x, shoot_y), 15, (0, 255, 255), 3)
    
    last_gesture_state = current_gesture
    
    if gesture_cooldown > 0:
        gesture_cooldown -= 1
    
    # Update and draw shooting strands (behind webs)
    for strand in shooting_strands:
        strand.update()
        strand.draw(frame)
    shooting_strands = [s for s in shooting_strands if s.active]
    
    # Update and draw web effects
    for web in web_effects:
        web.update()
        web.draw(frame)
    web_effects = [web for web in web_effects if web.active]
    
    # HUD
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (0, 0), (450, 130), (0, 0, 0), -1)
    cv2.addWeighted(hud_overlay, 0.3, frame, 0.7, 0, frame)
    
    cv2.putText(frame, "SPIDER-MAN WEB SHOOTER", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    cv2.putText(frame, f"Active Webs: {len(web_effects)}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if current_gesture:
        cv2.putText(frame, "GESTURE DETECTED!", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if gesture_cooldown > 0:
            bar_width = int((gesture_cooldown / 25) * 220)
            cv2.rectangle(frame, (10, 110), (230, 120), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 110), (10 + bar_width, 120), (0, 165, 255), -1)
    
    sound_status = "ON" if (sound_available and sound_enabled) else "OFF"
    sound_color = (0, 255, 0) if (sound_available and sound_enabled) else (0, 0, 255)
    cv2.putText(frame, f"Sound: {sound_status} (Press 's')", (frame_width - 300, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, sound_color, 2)
    
    cv2.imshow('Spider-Man Web Shooter', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        sound_enabled = not sound_enabled
        print(f"Sound {'enabled' if sound_enabled else 'disabled'}")

cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.mixer.quit()
print("\n🕷️  Thanks for using Spider-Man Web Shooter! 🕷️")