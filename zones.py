# Developer: tubakhxn

import cv2
import numpy as np


class Zone:
    def __init__(self, name, points, color, zone_type="generic", alpha=0.32):
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.color = color
        self.zone_type = zone_type
        self.alpha = alpha

    def contains_point(self, point):
        x, y = float(point[0]), float(point[1])
        return cv2.pointPolygonTest(self.points, (x, y), False) >= 0

    def draw(self, frame):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.points], self.color)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)
        cv2.polylines(frame, [self.points], isClosed=True, color=self.color,
                       thickness=2, lineType=cv2.LINE_AA)

        label_x, label_y = self.points[0][0], self.points[0][1]
        text = self.name
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (label_x, label_y - th - 10), (label_x + tw + 8, label_y - 2),
                      (20, 20, 20), -1)
        cv2.putText(frame, text, (label_x + 4, label_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1, cv2.LINE_AA)
        return frame


class ZoneManager:
    COLOR_CROSSWALK = (50, 50, 235)
    COLOR_WAITING = (235, 160, 50)
    COLOR_NEAR_CROSSWALK = (0, 165, 255)
    COLOR_STOP = (60, 200, 60)

    def __init__(self, frame_width, frame_height):
        self.w = frame_width
        self.h = frame_height
        self.zones = []
        self._build_default_layout()

    def _build_default_layout(self):
        w, h = self.w, self.h

        crosswalk = Zone(
            "crosswalk",
            [(int(w * 0.28), int(h * 0.52)), (int(w * 0.72), int(h * 0.52)),
             (int(w * 0.96), int(h * 0.88)), (int(w * 0.04), int(h * 0.88))],
            self.COLOR_CROSSWALK, zone_type="crosswalk", alpha=0.28,
        )

        waiting_left = Zone(
            "waiting_zone",
            [(int(w * 0.01), int(h * 0.42)), (int(w * 0.27), int(h * 0.42)),
             (int(w * 0.19), int(h * 0.60)), (int(w * 0.01), int(h * 0.60))],
            self.COLOR_WAITING, zone_type="waiting", alpha=0.30,
        )

        waiting_right = Zone(
            "waiting_zone",
            [(int(w * 0.73), int(h * 0.42)), (int(w * 0.99), int(h * 0.42)),
             (int(w * 0.99), int(h * 0.60)), (int(w * 0.81), int(h * 0.60))],
            self.COLOR_WAITING, zone_type="waiting", alpha=0.30,
        )

        near_crosswalk = Zone(
            "near_crosswalk",
            [(int(w * 0.30), int(h * 0.38)), (int(w * 0.70), int(h * 0.38)),
             (int(w * 0.72), int(h * 0.52)), (int(w * 0.28), int(h * 0.52))],
            self.COLOR_NEAR_CROSSWALK, zone_type="near_crosswalk", alpha=0.24,
        )

        vehicle_stop = Zone(
            "vehicle_stop_zone",
            [(int(w * 0.04), int(h * 0.88)), (int(w * 0.96), int(h * 0.88)),
             (int(w * 1.00), int(h * 1.00)), (int(w * 0.00), int(h * 1.00))],
            self.COLOR_STOP, zone_type="stop", alpha=0.18,
        )

        self.zones = [crosswalk, waiting_left, waiting_right, near_crosswalk, vehicle_stop]

    def get_zone(self, zone_type):
        for z in self.zones:
            if z.zone_type == zone_type:
                return z
        return None

    def get_zones(self, zone_type):
        return [z for z in self.zones if z.zone_type == zone_type]

    def zones_containing(self, point):
        return [z.name for z in self.zones if z.contains_point(point)]

    def is_in_crosswalk(self, point):
        z = self.get_zone("crosswalk")
        return z is not None and z.contains_point(point)

    def is_in_near_crosswalk(self, point):
        return any(z.contains_point(point) for z in self.get_zones("near_crosswalk"))

    def is_in_waiting_zone(self, point):
        return any(z.contains_point(point) for z in self.get_zones("waiting"))

    def is_in_stop_zone(self, point):
        z = self.get_zone("stop")
        return z is not None and z.contains_point(point)

    def draw_all(self, frame):
        for zone in self.zones:
            zone.draw(frame)
        return frame
