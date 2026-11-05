"""FRUIT SLASH - Game chém trái cây bằng camera và cử chỉ hai bàn tay.

Chạy trên Windows với Python 3.11 64-bit:
    py -3.11 -m pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import pygame


# ---------------------------------------------------------------------------
# CẤU HÌNH
# ---------------------------------------------------------------------------
WINDOW_W, WINDOW_H = 1360, 840
FPS = 60
CAMERA_INDEX = 0
CAMERA_W, CAMERA_H = 640, 480

BG = (7, 13, 27)
PANEL = (15, 26, 46)
PANEL_LIGHT = (20, 36, 60)
WHITE = (239, 247, 255)
MUTED = (145, 166, 195)
CYAN = (75, 222, 241)
GREEN = (87, 225, 164)
YELLOW = (255, 207, 103)
RED = (255, 91, 112)
PURPLE = (167, 139, 250)

CAMERA_PANEL = pygame.Rect(24, 124, 386, 322)
CAMERA_VIEW = pygame.Rect(40, 164, 354, 234)
STATUS_PANEL = pygame.Rect(24, 462, 386, 96)
HELP_PANEL = pygame.Rect(24, 574, 386, 218)
GAME_RECT = pygame.Rect(432, 124, 904, 668)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def draw_text(surface, text, size, color, position, bold=False, anchor="topleft"):
    font = pygame.font.SysFont("segoeui", size, bold=bold)
    image = font.render(str(text), True, color)
    rect = image.get_rect()
    setattr(rect, anchor, position)
    surface.blit(image, rect)
    return rect


def panel(surface, rect, fill=PANEL, border=(39, 60, 91), radius=18, width=1):
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surface, border, rect, width=width, border_radius=radius)


def segment_distance(px, py, ax, ay, bx, by):
    """Khoảng cách từ điểm P đến đoạn thẳng AB."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 0.0001:
        return math.hypot(px - ax, py - ay)
    t = clamp(((px - ax) * dx + (py - ay) * dy) / length_sq, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


# ---------------------------------------------------------------------------
# CAMERA VÀ NHẬN DIỆN
# ---------------------------------------------------------------------------
class CameraTracker:
    """Mở webcam, xác nhận người bằng Pose và theo dõi tối đa hai bàn tay."""

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.camera_ok = self.cap.isOpened()
        self.error = "" if self.camera_ok else (
            "Không mở được webcam. Hãy kiểm tra quyền camera hoặc CAMERA_INDEX."
        )

        self.pose_api = mp.solutions.pose
        self.hands_api = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils

        self.pose = self.pose_api.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self.hands = self.hands_api.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        self.preview = None
        self.person_detected = False
        self.hand_points: list[Optional[tuple[float, float]]] = [None, None]
        self.last_inference = 0.0
        self.inference_interval = 1.0 / 30.0

    def update(self, now):
        if not self.camera_ok:
            return self.preview, False, [None, None]

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.person_detected = False
            self.hand_points = [None, None]
            return self.preview, False, [None, None]

        # Lật ảnh như gương để thao tác tay tự nhiên hơn.
        frame = cv2.flip(frame, 1)

        if now - self.last_inference >= self.inference_interval:
            self.last_inference = now
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_result = self.pose.process(rgb)
            hands_result = self.hands.process(rgb)
            rgb.flags.writeable = True

            self.person_detected = pose_result.pose_landmarks is not None
            self.hand_points = [None, None]

            if pose_result.pose_landmarks:
                self.drawer.draw_landmarks(
                    frame,
                    pose_result.pose_landmarks,
                    self.pose_api.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.drawer.DrawingSpec(
                        color=(75, 220, 170), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=self.drawer.DrawingSpec(
                        color=(55, 155, 145), thickness=2, circle_radius=1
                    ),
                )

            if hands_result.multi_hand_landmarks:
                for index, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
                    self.drawer.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.hands_api.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.drawer.DrawingSpec(
                            color=(70, 225, 245), thickness=2, circle_radius=3
                        ),
                        connection_drawing_spec=self.drawer.DrawingSpec(
                            color=(105, 160, 255), thickness=2, circle_radius=2
                        ),
                    )

                    # Gán vị trí theo nhãn bàn tay để tránh đổi tay trái/phải giữa các frame.
                    slot = index
                    if hands_result.multi_handedness and index < len(hands_result.multi_handedness):
                        labels = hands_result.multi_handedness[index].classification
                        if labels:
                            slot = 0 if labels[0].label.lower() == "left" else 1

                    tip = hand_landmarks.landmark[8]  # đầu ngón trỏ
                    self.hand_points[slot] = (float(tip.x), float(tip.y))

        self.preview = frame
        return self.preview, self.person_detected, list(self.hand_points)

    def close(self):
        try:
            if self.cap:
                self.cap.release()
            self.hands.close()
            self.pose.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ĐỐI TƯỢNG GAME VÀ HIỆU ỨNG
# ---------------------------------------------------------------------------
FRUIT_STYLES = [
    ("TÁO", (238, 65, 82), (255, 159, 144)),
    ("CAM", (255, 157, 48), (255, 221, 125)),
    ("DƯA", (57, 188, 111), (245, 92, 125)),
    ("NHO", (151, 105, 224), (210, 178, 255)),
    ("CHANH", (205, 218, 69), (255, 245, 143)),
    ("DÂU", (237, 69, 105), (255, 183, 190)),
]


@dataclass
class Fruit:
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    style: tuple
    bomb: bool = False
    rotation: float = 0.0

    def update(self, gravity, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += gravity * dt
        self.rotation += self.vx * 0.65 * dt

    def draw(self, surface):
        x, y, r = int(self.x), int(self.y), self.radius
        if self.bomb:
            pygame.draw.circle(surface, (3, 7, 15), (x + 2, y + 4), r + 3)
            pygame.draw.circle(surface, (14, 19, 31), (x, y), r)
            pygame.draw.circle(surface, RED, (x, y), r, 3)
            pygame.draw.circle(surface, (92, 104, 126), (x - 8, y - 9), 7)
            pygame.draw.line(surface, (217, 171, 104), (x + 4, y - r + 1), (x + 16, y - r - 13), 5)
            pygame.draw.circle(surface, YELLOW, (x + 17, y - r - 14), 5)
            pygame.draw.circle(surface, WHITE, (x + 17, y - r - 14), 2)
            return

        name, color, inner = self.style
        pygame.draw.ellipse(surface, (3, 7, 15), (x - r + 3, y + r - 5, r * 2 - 2, 9))
        pygame.draw.circle(surface, color, (x, y), r)
        pygame.draw.circle(surface, inner, (x - r // 3, y - r // 3), max(5, r // 4))
        pygame.draw.circle(surface, WHITE, (x - r // 3, y - r // 3), max(2, r // 10))
        pygame.draw.arc(surface, (255, 255, 255), (x - r + 4, y - r + 4, r * 2 - 8, r * 2 - 8), 3.5, 5.2, 2)
        pygame.draw.line(surface, (104, 69, 43), (x, y - r + 2), (x + 3, y - r - 12), 4)
        pygame.draw.ellipse(surface, (80, 206, 117), (x + 1, y - r - 12, 20, 10))
        if name in ("DÂU", "CHANH"):
            for ox, oy in [(-8, 0), (5, 7), (9, -5), (-3, 12)]:
                pygame.draw.circle(surface, (255, 239, 185), (x + ox, y + oy), 2)


@dataclass
class SliceHalf:
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    color: tuple
    inside: tuple
    angle: float
    spin: float = 0.0
    life: float = 1.0

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 0.24 * dt
        self.spin += self.vx * 0.018 * dt
        self.life -= 0.018 * dt

    def draw(self, surface):
        if self.life <= 0:
            return
        a = self.angle + self.spin
        points = [(int(self.x), int(self.y))]
        for i in range(15):
            angle = a + math.pi * i / 14
            points.append((
                int(self.x + math.cos(angle) * self.radius),
                int(self.y + math.sin(angle) * self.radius),
            ))
        pygame.draw.polygon(surface, self.color, points)
        pygame.draw.polygon(surface, (255, 241, 226), points, 2)
        p1 = (int(self.x + math.cos(a) * self.radius), int(self.y + math.sin(a) * self.radius))
        p2 = (int(self.x + math.cos(a + math.pi) * self.radius), int(self.y + math.sin(a + math.pi) * self.radius))
        pygame.draw.line(surface, self.inside, p1, p2, 5)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: tuple
    life: float
    size: int

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 0.16 * dt
        self.life -= 0.025 * dt

    def draw(self, surface):
        if self.life > 0:
            radius = max(1, int(self.size * clamp(self.life, 0, 1)))
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), radius)


# ---------------------------------------------------------------------------
# ỨNG DỤNG
# ---------------------------------------------------------------------------
class FruitSlashApp:
    MENU, COUNTDOWN, PLAYING, PAUSED, GAMEOVER = (
        "MENU", "COUNTDOWN", "PLAYING", "PAUSED", "GAMEOVER"
    )

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("FRUIT SLASH | Chém trái cây bằng camera")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.running = True

        self.camera = CameraTracker(CAMERA_INDEX)
        self.preview = None
        self.person_detected = False
        self.hand_points_norm = [None, None]

        self.state = self.MENU
        self.score = 0
        self.lives = 3
        self.level = 1
        self.started_at = 0.0
        self.countdown_at = 0.0
        self.next_spawn = 0.0

        self.fruits: list[Fruit] = []
        self.halves: list[SliceHalf] = []
        self.particles: list[Particle] = []
        self.hand_positions: list[Optional[tuple[float, float]]] = [None, None]
        self.previous_hands: list[Optional[tuple[float, float]]] = [None, None]
        self.trails: list[list[tuple[float, float, float]]] = [[], []]

        self.start_button = pygame.Rect(0, 0, 344, 82)
        self.start_button.center = (GAME_RECT.centerx, GAME_RECT.centery + 72)
        self.dwell_started: Optional[float] = None
        self.dwell_seconds = 0.95
        self.toast = ""
        self.toast_until = 0.0
        self.pause_at: Optional[float] = None

    def reset_game(self):
        self.score = 0
        self.lives = 3
        self.level = 1
        self.fruits.clear()
        self.halves.clear()
        self.particles.clear()
        self.previous_hands = [None, None]
        self.trails = [[], []]
        self.state = self.COUNTDOWN
        self.countdown_at = time.perf_counter()
        self.dwell_started = None
        self.pause_at = None

    def toast_message(self, text, seconds=2.2):
        self.toast = text
        self.toast_until = time.perf_counter() + seconds

    def map_hand(self, point):
        if point is None:
            return None
        nx, ny = point
        return (
            GAME_RECT.left + clamp(nx, 0.0, 1.0) * GAME_RECT.width,
            GAME_RECT.top + clamp(ny, 0.0, 1.0) * GAME_RECT.height,
        )

    def update_hands(self, now):
        self.hand_positions = [self.map_hand(p) for p in self.hand_points_norm]
        for i in range(2):
            current = self.hand_positions[i]
            previous = self.previous_hands[i]
            if current is None:
                self.previous_hands[i] = None
                self.trails[i] = [p for p in self.trails[i] if now - p[2] < 0.2]
                continue

            self.trails[i].append((current[0], current[1], now))
            self.trails[i] = [p for p in self.trails[i] if now - p[2] < 0.22]

            if self.state == self.PLAYING and self.person_detected and previous:
                movement = math.hypot(current[0] - previous[0], current[1] - previous[1])
                if movement >= 7.0:
                    self.check_slash(previous, current)

            self.previous_hands[i] = current

    def check_slash(self, start, end):
        hit_list = []
        for fruit in self.fruits:
            distance = segment_distance(
                fruit.x, fruit.y, start[0], start[1], end[0], end[1]
            )
            if distance <= fruit.radius + 14:
                hit_list.append(fruit)

        for fruit in hit_list:
            if fruit not in self.fruits:
                continue
            self.fruits.remove(fruit)
            if fruit.bomb:
                self.explode(fruit.x, fruit.y, RED, 44)
                self.state = self.GAMEOVER
                self.toast_message("BOOM! Bạn đã chém trúng bom.", 3.0)
                return

            self.score += 10
            self.slice_fruit(fruit, start, end)
            self.explode(fruit.x, fruit.y, fruit.style[1], 20)

    def slice_fruit(self, fruit, start, end):
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        side = angle + math.pi / 2
        _, color, inside = fruit.style
        for sign in (-1, 1):
            self.halves.append(SliceHalf(
                fruit.x, fruit.y,
                fruit.vx + math.cos(side) * 3.1 * sign,
                fruit.vy - 1.8 + math.sin(side) * 3.1 * sign,
                fruit.radius, color, inside,
                angle + (math.pi if sign > 0 else 0),
            ))

    def explode(self, x, y, color, amount=20):
        for _ in range(amount):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(1.5, 7.0)
            self.particles.append(Particle(
                x, y,
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                random.choice([color, WHITE, YELLOW, CYAN]),
                random.uniform(0.65, 1.0),
                random.randint(3, 7),
            ))

    def spawn_wave(self, now, elapsed):
        self.level = 1 + int(elapsed // 18)
        fruit_count = min(1 + int(elapsed // 15), 6)
        speed_bonus = min(self.level - 1, 9) * 0.75

        for _ in range(fruit_count):
            radius = random.randint(27, 37)
            x = random.randint(GAME_RECT.left + 55, GAME_RECT.right - 55)
            y = GAME_RECT.bottom + radius + random.randint(0, 24)
            vx = random.uniform(-2.5, 2.5)
            vy = -random.uniform(12.0, 15.0) - speed_bonus
            self.fruits.append(Fruit(
                x, y, vx, vy, radius, random.choice(FRUIT_STYLES), False
            ))

        # Ban đầu bom hiếm; xác suất tăng dần nhưng được giới hạn.
        bomb_chance = min(0.08 + max(0.0, elapsed - 6.0) * 0.006, 0.30)
        if elapsed > 6 and random.random() < bomb_chance:
            radius = 31
            x = random.randint(GAME_RECT.left + 60, GAME_RECT.right - 60)
            y = GAME_RECT.bottom + radius + random.randint(0, 30)
            vx = random.uniform(-2.0, 2.0)
            vy = -random.uniform(12.5, 15.5) - speed_bonus
            self.fruits.append(Fruit(x, y, vx, vy, radius, (), True))

        interval = max(0.48, 1.48 - elapsed * 0.012)
        self.next_spawn = now + interval

    def update_game(self, now, dt):
        if self.state == self.COUNTDOWN:
            if now - self.countdown_at >= 3.0:
                self.state = self.PLAYING
                self.started_at = now
                self.next_spawn = now + 0.3
            return

        if self.state != self.PLAYING:
            return

        elapsed = now - self.started_at
        self.level = 1 + int(elapsed // 18)
        gravity = 0.22 + min(self.level - 1, 9) * 0.012

        if now >= self.next_spawn:
            self.spawn_wave(now, elapsed)

        for fruit in self.fruits[:]:
            fruit.update(gravity, dt)
            if fruit.x < GAME_RECT.left + fruit.radius or fruit.x > GAME_RECT.right - fruit.radius:
                fruit.vx *= -1
            if fruit.y > GAME_RECT.bottom + fruit.radius + 25:
                self.fruits.remove(fruit)
                if not fruit.bomb:
                    self.lives -= 1
                    if self.lives <= 0:
                        self.state = self.GAMEOVER
                        self.toast_message("Bạn đã bỏ lỡ quá nhiều trái cây.", 3.0)
                        break

    def update_effects(self, dt):
        for particle in self.particles[:]:
            particle.update(dt)
            if particle.life <= 0:
                self.particles.remove(particle)
        for half in self.halves[:]:
            half.update(dt)
            if half.life <= 0 or half.y > GAME_RECT.bottom + 100:
                self.halves.remove(half)

    def update_button_dwell(self, now):
        if self.state not in (self.MENU, self.GAMEOVER) or not self.person_detected:
            self.dwell_started = None
            return 0.0, False

        hovered = any(
            p is not None and self.start_button.collidepoint(p)
            for p in self.hand_positions
        )
        if not hovered:
            self.dwell_started = None
            return 0.0, False

        if self.dwell_started is None:
            self.dwell_started = now
        progress = clamp((now - self.dwell_started) / self.dwell_seconds, 0.0, 1.0)
        if progress >= 1.0:
            self.reset_game()
            return 1.0, True
        return progress, True

    # -----------------------------------------------------------------------
    # VẼ GIAO DIỆN
    # -----------------------------------------------------------------------
    def draw_background(self):
        self.screen.fill(BG)
        for i in range(10):
            x = (i * 173 + 65) % WINDOW_W
            pygame.draw.circle(self.screen, (15, 29, 50), (x, 94 + (i % 3) * 13), 3)
        pygame.draw.line(self.screen, (29, 47, 73), (24, 100), (WINDOW_W - 24, 100), 1)

    def draw_header(self):
        pygame.draw.circle(self.screen, (24, 92, 124), (54, 51), 25)
        pygame.draw.circle(self.screen, CYAN, (54, 51), 25, 2)
        draw_text(self.screen, "✦", 25, WHITE, (54, 50), bold=True, anchor="center")
        draw_text(self.screen, "FRUIT SLASH", 26, WHITE, (91, 29), bold=True)
        draw_text(self.screen, "CAMERA MOTION GAME  •  HAND TRACKING", 11, MUTED, (93, 63), bold=True)
        pygame.draw.circle(self.screen, GREEN if self.camera.camera_ok else RED, (1268, 45), 5)
        draw_text(self.screen, "AI VISION", 12, GREEN if self.camera.camera_ok else RED, (1335, 36), bold=True, anchor="topright")
        draw_text(self.screen, "XỬ LÝ CAMERA TẠI MÁY", 10, MUTED, (1335, 61), anchor="topright")

    def draw_camera(self):
        panel(self.screen, CAMERA_PANEL)
        draw_text(self.screen, "CAMERA TRỰC TIẾP", 14, WHITE, (42, 135), bold=True)
        status = "ĐÃ NHẬN DIỆN NGƯỜI" if self.person_detected else "ĐANG TÌM NGƯỜI..."
        draw_text(self.screen, status, 9, GREEN if self.person_detected else YELLOW, (394, 139), bold=True, anchor="topright")

        pygame.draw.rect(self.screen, (4, 8, 16), CAMERA_VIEW, border_radius=12)
        if self.preview is not None:
            try:
                frame = cv2.resize(self.preview, CAMERA_VIEW.size, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = pygame.image.frombuffer(rgb.tobytes(), CAMERA_VIEW.size, "RGB")
                self.screen.blit(image, CAMERA_VIEW.topleft)
            except Exception:
                draw_text(self.screen, "Không hiển thị được camera", 12, MUTED, CAMERA_VIEW.center, anchor="center")
        else:
            draw_text(self.screen, "CAMERA CHƯA SẴN SÀNG", 14, MUTED, CAMERA_VIEW.center, bold=True, anchor="center")
            draw_text(self.screen, "Kiểm tra webcam và quyền truy cập", 10, MUTED, (CAMERA_VIEW.centerx, CAMERA_VIEW.centery + 24), anchor="center")
        pygame.draw.rect(self.screen, (56, 85, 119), CAMERA_VIEW, width=2, border_radius=12)

        panel(self.screen, STATUS_PANEL)
        dot_color = GREEN if self.person_detected else YELLOW
        pygame.draw.circle(self.screen, dot_color, (48, 490), 6)
        draw_text(self.screen, "Đã phát hiện người" if self.person_detected else "Chưa thấy người trong khung hình", 12, WHITE, (64, 481), bold=True)
        count = sum(1 for p in self.hand_points_norm if p is not None)
        draw_text(self.screen, f"BÀN TAY ĐƯỢC THEO DÕI: {count}/2", 11, CYAN, (42, 520), bold=True)
        draw_text(self.screen, "Đưa thân trên và hai tay vào vùng camera.", 10, MUTED, (42, 540))

        panel(self.screen, HELP_PANEL)
        draw_text(self.screen, "CÁCH ĐIỀU KHIỂN", 13, CYAN, (42, 592), bold=True)
        help_lines = [
            ("01", "Đưa ngón trỏ vào nút BẮT ĐẦU và giữ 1 giây."),
            ("02", "Vung tay để đường di chuyển cắt qua trái cây."),
            ("03", "Có thể dùng một tay hoặc cả hai tay."),
            ("04", "Tránh bom. Bỏ lỡ trái cây sẽ mất một mạng."),
        ]
        for i, (number, text) in enumerate(help_lines):
            y = 626 + i * 34
            draw_text(self.screen, number, 10, YELLOW, (44, y), bold=True)
            draw_text(self.screen, text, 10, WHITE, (72, y))

        draw_text(self.screen, "P: TẠM DỪNG     ESC: THOÁT", 9, MUTED, (42, 766))

    def draw_game_background(self):
        panel(self.screen, GAME_RECT, fill=(10, 20, 39), border=(42, 68, 101), radius=22)
        for x in range(GAME_RECT.left + 30, GAME_RECT.right, 48):
            pygame.draw.line(self.screen, (17, 33, 55), (x, GAME_RECT.top + 8), (x, GAME_RECT.bottom - 8), 1)
        for y in range(GAME_RECT.top + 25, GAME_RECT.bottom, 48):
            pygame.draw.line(self.screen, (17, 33, 55), (GAME_RECT.left + 8, y), (GAME_RECT.right - 8, y), 1)
        pygame.draw.line(self.screen, (31, 69, 87), (GAME_RECT.left + 20, GAME_RECT.bottom - 22), (GAME_RECT.right - 20, GAME_RECT.bottom - 22), 2)

    def draw_hud(self, now):
        hud = pygame.Rect(GAME_RECT.x + 18, GAME_RECT.y + 16, GAME_RECT.width - 36, 62)
        panel(self.screen, hud, fill=(16, 33, 56), border=(41, 69, 101), radius=14)
        draw_text(self.screen, "ĐIỂM", 9, MUTED, (hud.x + 18, hud.y + 8), bold=True)
        draw_text(self.screen, f"{self.score:04d}", 23, WHITE, (hud.x + 18, hud.y + 25), bold=True)
        draw_text(self.screen, "CẤP ĐỘ", 9, MUTED, (hud.x + 185, hud.y + 8), bold=True)
        draw_text(self.screen, str(self.level), 23, CYAN, (hud.x + 185, hud.y + 25), bold=True)
        draw_text(self.screen, "MẠNG", 9, MUTED, (hud.right - 142, hud.y + 8), bold=True)
        for i in range(3):
            color = RED if i < self.lives else (61, 73, 91)
            hx = hud.right - 125 + i * 29
            pygame.draw.circle(self.screen, color, (hx - 5, hud.y + 36), 7)
            pygame.draw.circle(self.screen, color, (hx + 5, hud.y + 36), 7)
            pygame.draw.polygon(self.screen, color, [(hx - 12, hud.y + 38), (hx + 12, hud.y + 38), (hx, hud.y + 53)])
        if self.state == self.PLAYING:
            elapsed = max(0, int(now - self.started_at))
            draw_text(self.screen, f"{elapsed // 60:02d}:{elapsed % 60:02d}", 12, MUTED, (hud.centerx + 4, hud.y + 37), anchor="center")

    def draw_hands(self, now):
        for trail in self.trails:
            pts = [(int(x), int(y)) for x, y, t in trail if now - t < 0.20]
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, (62, 208, 240), False, pts, 5)
                for p in pts[-3:]:
                    pygame.draw.circle(self.screen, (155, 246, 255), p, 5)
        for point in self.hand_positions:
            if point and GAME_RECT.collidepoint(point):
                p = (int(point[0]), int(point[1]))
                pygame.draw.circle(self.screen, CYAN, p, 17, 2)
                pygame.draw.circle(self.screen, WHITE, p, 4)

    def draw_button(self, progress=0.0, hover=False, enabled=True, label="BẮT ĐẦU TRÒ CHƠI"):
        if enabled and hover:
            fill, border_color, text_color = (26, 103, 132), CYAN, WHITE
        elif enabled:
            fill, border_color, text_color = (19, 77, 105), (68, 190, 216), WHITE
        else:
            fill, border_color, text_color = (48, 60, 78), (75, 88, 108), (159, 172, 190)

        pygame.draw.rect(self.screen, fill, self.start_button, border_radius=19)
        pygame.draw.rect(self.screen, border_color, self.start_button, width=2, border_radius=19)
        draw_text(self.screen, label, 20, text_color, self.start_button.center, bold=True, anchor="center")
        if enabled and progress > 0:
            bar = pygame.Rect(self.start_button.x + 15, self.start_button.bottom - 8, int((self.start_button.width - 30) * progress), 4)
            pygame.draw.rect(self.screen, YELLOW, bar, border_radius=3)

    def draw_menu(self, progress, hovered):
        draw_text(self.screen, "SẴN SÀNG CHƯA?", 34, WHITE, (GAME_RECT.centerx, GAME_RECT.centery - 126), bold=True, anchor="center")
        draw_text(self.screen, "Dùng chính chuyển động cơ thể để chém trái cây", 15, MUTED, (GAME_RECT.centerx, GAME_RECT.centery - 82), anchor="center")
        ready = self.person_detected
        self.draw_button(progress, hovered, ready, "BẮT ĐẦU TRÒ CHƠI" if ready else "ĐANG CHỜ NHẬN DIỆN NGƯỜI")
        instruction = "Đưa ngón trỏ vào nút và giữ yên khoảng 1 giây" if ready else "Hãy đứng trước camera để hệ thống nhận diện bạn"
        draw_text(self.screen, instruction, 11, CYAN if ready else YELLOW, (GAME_RECT.centerx, self.start_button.bottom + 20), anchor="center")

        cards = [
            ("ĐIỀU KHIỂN 2 TAY", "Mỗi bàn tay là một lưỡi dao ảo", CYAN),
            ("TĂNG ĐỘ KHÓ", "Trái cây nhanh và xuất hiện dày hơn", GREEN),
            ("CẨN THẬN BOM", "Chém trúng bom là kết thúc", RED),
        ]
        card_w, gap = 250, 20
        total = card_w * 3 + gap * 2
        start_x = GAME_RECT.centerx - total // 2
        for i, (title, desc, color) in enumerate(cards):
            rect = pygame.Rect(start_x + i * (card_w + gap), GAME_RECT.bottom - 94, card_w, 68)
            panel(self.screen, rect, fill=(15, 32, 54), border=(36, 61, 89), radius=12)
            draw_text(self.screen, title, 11, color, (rect.x + 13, rect.y + 12), bold=True)
            draw_text(self.screen, desc, 9, MUTED, (rect.x + 13, rect.y + 38))

    def draw_gameover(self):
        shade = pygame.Surface(GAME_RECT.size, pygame.SRCALPHA)
        shade.fill((3, 7, 16, 205))
        self.screen.blit(shade, GAME_RECT.topleft)
        draw_text(self.screen, "TRÒ CHƠI KẾT THÚC", 34, RED, (GAME_RECT.centerx, GAME_RECT.centery - 115), bold=True, anchor="center")
        draw_text(self.screen, f"ĐIỂM CỦA BẠN: {self.score}", 23, WHITE, (GAME_RECT.centerx, GAME_RECT.centery - 60), bold=True, anchor="center")
        self.draw_button(0.0, False, self.person_detected, "CHƠI LẠI" if self.person_detected else "ĐANG CHỜ NHẬN DIỆN NGƯỜI")
        hint = "Đưa ngón trỏ vào nút để chơi lại" if self.person_detected else "Đứng trước camera để chơi lại"
        draw_text(self.screen, hint, 11, MUTED, (GAME_RECT.centerx, self.start_button.bottom + 20), anchor="center")

    def draw_overlay(self, now):
        if self.state == self.COUNTDOWN:
            remaining = max(1, 3 - int(now - self.countdown_at))
            draw_text(self.screen, str(remaining), 100, YELLOW, GAME_RECT.center, bold=True, anchor="center")
            draw_text(self.screen, "CHUẨN BỊ CHÉM!", 16, WHITE, (GAME_RECT.centerx, GAME_RECT.centery + 75), bold=True, anchor="center")
        elif self.state == self.PAUSED:
            shade = pygame.Surface(GAME_RECT.size, pygame.SRCALPHA)
            shade.fill((3, 7, 16, 185))
            self.screen.blit(shade, GAME_RECT.topleft)
            draw_text(self.screen, "ĐÃ TẠM DỪNG", 36, WHITE, GAME_RECT.center, bold=True, anchor="center")
            draw_text(self.screen, "Nhấn P để tiếp tục", 14, MUTED, (GAME_RECT.centerx, GAME_RECT.centery + 55), anchor="center")
        elif self.state == self.GAMEOVER:
            self.draw_gameover()

    def draw_toast(self, now):
        if self.toast and now < self.toast_until:
            rect = pygame.Rect(GAME_RECT.centerx - 220, GAME_RECT.bottom - 58, 440, 36)
            panel(self.screen, rect, fill=(75, 27, 42), border=(181, 65, 83), radius=10)
            draw_text(self.screen, self.toast, 11, WHITE, rect.center, bold=True, anchor="center")

    def draw(self, now, progress, hovered):
        self.draw_background()
        self.draw_header()
        self.draw_camera()
        self.draw_game_background()
        if self.state in (self.COUNTDOWN, self.PLAYING, self.PAUSED, self.GAMEOVER):
            self.draw_hud(now)

        for fruit in self.fruits:
            fruit.draw(self.screen)
        for half in self.halves:
            half.draw(self.screen)
        for particle in self.particles:
            particle.draw(self.screen)
        self.draw_hands(now)

        if self.state == self.MENU:
            self.draw_menu(progress, hovered)
        else:
            self.draw_overlay(now)

        self.draw_toast(now)
        draw_text(self.screen, "SPACE: BẮT ĐẦU / CHƠI LẠI     P: TẠM DỪNG     ESC: THOÁT", 9, (105, 126, 155), (WINDOW_W - 26, WINDOW_H - 15), anchor="bottomright")
        draw_text(self.screen, "FRUIT SLASH  •  CAMERA & AI XỬ LÝ TRỰC TIẾP TRÊN MÁY", 9, (105, 126, 155), (26, WINDOW_H - 15), anchor="bottomleft")
        pygame.display.flip()

    # -----------------------------------------------------------------------
    # VÒNG LẶP
    # -----------------------------------------------------------------------
    def run(self):
        try:
            while self.running:
                elapsed_ms = self.clock.tick(FPS)
                dt = clamp(elapsed_ms / (1000.0 / FPS), 0.25, 2.5)
                now = time.perf_counter()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE and self.state in (self.MENU, self.GAMEOVER):
                            self.reset_game()
                        elif event.key == pygame.K_p:
                            if self.state == self.PLAYING:
                                self.state = self.PAUSED
                                self.pause_at = now
                                self.previous_hands = [None, None]
                            elif self.state == self.PAUSED:
                                paused = now - self.pause_at if self.pause_at else 0.0
                                self.started_at += paused
                                self.next_spawn += paused
                                self.state = self.PLAYING
                                self.pause_at = None
                                self.previous_hands = [None, None]

                self.preview, self.person_detected, self.hand_points_norm = self.camera.update(now)
                self.update_hands(now)
                progress, hovered = self.update_button_dwell(now)
                self.update_game(now, dt)
                self.update_effects(dt)
                self.draw(now, progress, hovered)
        finally:
            self.camera.close()
            pygame.quit()


def main():
    app = None
    try:
        app = FruitSlashApp()
        if not app.camera.camera_ok:
            print(app.camera.error)
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("\nFRUIT SLASH gặp lỗi:")
        print(f"{type(exc).__name__}: {exc}")
        print("Hãy kiểm tra Python 3.11 64-bit và chạy run.bat để cài thư viện.")
        try:
            input("Nhấn Enter để đóng...")
        except EOFError:
            pass
    finally:
        if app is not None and app.running:
            try:
                app.camera.close()
                pygame.quit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
