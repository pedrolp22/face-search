# scripts/webcam_recognition.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import json
import faiss
import face_recognition
import time
import math
from collections import Counter, deque

INDEX_PATH = "../embeddings/faiss.index"
METADATA_PATH = "../embeddings/metadata.json"

# require distance <= this to be considered recognized/displayed
DISPLAY_THRESHOLD = 0.25

# simple temporal smoothing (per tracked face)
BUFFER_SIZE = 5
MATCH_DISTANCE_PX = 60
MAX_TRACK_MISSING = 10

# load index + metadata
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# open webcam (0 is default camera)
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Could not open webcam")

# optional: reduce processing resolution for speed
SCALE = 0.5  # process at 50% size; set to 1.0 if you want full resolution

tracks = []
next_track_id = 1
frame_idx = 0


def box_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def center_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def majority_label(history):
    if not history:
        return "Unidentified"
    return Counter(history).most_common(1)[0][0]


try:
    while True:
        frame_idx += 1
        t0 = time.time()
        ret, frame = video.read()
        if not ret:
            break

        # optionally resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        # Ensure contiguous RGB for dlib/face_recognition.
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # detect face locations on the small frame
        face_locations = face_recognition.face_locations(rgb_small, model="hog")

        names = []
        boxes = []

        if face_locations:
            try:
                # explicit named arguments - avoid ambiguous call signature
                encodings = face_recognition.face_encodings(
                    rgb_small,
                    known_face_locations=face_locations,
                    num_jitters=0
                )
            except Exception as e:
                # don't crash on a frame; print debug and skip encodings
                print("Warning: face_encodings failed on this frame:", e)
                encodings = []

            for i, (loc_top, loc_right, loc_bottom, loc_left) in enumerate(face_locations):
                # scale box coords back to original frame size
                top = int(loc_top / SCALE)
                right = int(loc_right / SCALE)
                bottom = int(loc_bottom / SCALE)
                left = int(loc_left / SCALE)

                # clip to frame bounds to avoid off-screen boxes
                h, w = frame.shape[:2]
                top = max(0, min(top, h - 1))
                bottom = max(0, min(bottom, h - 1))
                left = max(0, min(left, w - 1))
                right = max(0, min(right, w - 1))

                name = "Unknown"
                distance = None

                if i < len(encodings):
                    # search index (ensure dtype float32)
                    q = np.array([encodings[i]]).astype("float32")
                    distances, indices = index.search(q, 1)
                    distance = float(distances[0][0])
                    idx = int(indices[0][0])

                    if distance <= DISPLAY_THRESHOLD:
                        name = metadata[idx]["person"]

                boxes.append((left, top, right, bottom))
                names.append((name, distance))

        # match detections to tracks for temporal smoothing
        unmatched_tracks = {t["id"] for t in tracks}
        matched = []

        for (left, top, right, bottom), (name, distance) in zip(boxes, names):
            bbox = (left, top, right, bottom)
            center = box_center(bbox)

            best_track = None
            best_dist = None
            for t in tracks:
                if t["id"] not in unmatched_tracks:
                    continue
                d = center_distance(center, t["center"])
                if d <= MATCH_DISTANCE_PX and (best_dist is None or d < best_dist):
                    best_dist = d
                    best_track = t

            is_recognized = distance is not None and distance <= DISPLAY_THRESHOLD and name != "Unknown"
            label_now = name if is_recognized else "Unidentified"

            if best_track is None:
                # new track
                track = {
                    "id": next_track_id,
                    "center": center,
                    "box": bbox,
                    "history": deque(maxlen=BUFFER_SIZE),
                    "last_seen": frame_idx,
                }
                next_track_id += 1
                tracks.append(track)
                best_track = track
            else:
                unmatched_tracks.remove(best_track["id"])
                best_track["center"] = center
                best_track["box"] = bbox
                best_track["last_seen"] = frame_idx

            best_track["history"].append(label_now)
            smoothed_label = majority_label(best_track["history"])
            matched.append((bbox, smoothed_label, distance))

        # drop stale tracks
        tracks = [t for t in tracks if frame_idx - t["last_seen"] <= MAX_TRACK_MISSING]

        # draw results on original frame
        for (x1, y1, x2, y2), smoothed_label, distance in matched:
            if smoothed_label != "Unidentified":
                box_color = (255, 255, 255)
                label = f"{smoothed_label} ({distance:.2f})" if distance is not None else smoothed_label
                text_color = (0, 0, 0)
            else:
                box_color = (0, 0, 255)  # red for unidentified
                label = "Unidentified"
                text_color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                frame,
                (x1, y2 + 4),
                (x1 + w + 6, y2 + 4 + h + 6),
                box_color,
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 3, y2 + h + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                1,
            )

        # show fps
        fps = 1.0 / (time.time() - t0) if (time.time() - t0) > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Face Recognition (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    video.release()
    cv2.destroyAllWindows()
