import cv2
import numpy as np
import time
from collections import deque

# ---------- CONFIG ----------
PROTO_FILE = "pose_deploy_linevec.prototxt"
WEIGHTS_FILE = "pose_iter_440000.caffemodel"

# Model input size (smaller = faster but less accurate)
IN_WIDTH = 368
IN_HEIGHT = 368
# Threshold to consider a keypoint detected
KEYPOINT_THRESHOLD = 0.1

# Pairs of keypoints for skeleton lines (COCO format)
POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4),
    (5, 6), (6, 7), (1, 8), (8, 9),
    (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]

# Number of body parts for COCO
N_POINTS = 18

# Tracking parameters
MAX_DISAPPEARED_FRAMES = 30  # after how many unseen frames remove a track
MAX_MATCH_DIST = 75  # pixels (to match centroids between frames)

# ---------- SIMPLE TRACKER ----------
class Track:
    def __init__(self, track_id, centroid, keypoints):
        self.id = track_id
        self.centroid = centroid
        self.keypoints = keypoints
        self.disappeared = 0
        self.history = deque(maxlen=30)
        self.history.append(centroid)

class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        """
        detections: list of dict { 'centroid': (x,y), 'keypoints': list of points or None }
        Returns list of tracks after update.
        """
        if len(self.tracks) == 0:
            for d in detections:
                t = Track(self.next_id, d['centroid'], d['keypoints'])
                self.next_id += 1
                self.tracks.append(t)
            return self.tracks

        # Build distance matrix between existing tracks and new detections
        distances = []
        for t in self.tracks:
            row = []
            for d in detections:
                dx = t.centroid[0] - d['centroid'][0]
                dy = t.centroid[1] - d['centroid'][1]
                row.append(np.hypot(dx, dy))
            distances.append(row)
        distances = np.array(distances)

        assigned_tracks = set()
        assigned_dets = set()

        if distances.size > 0:
            # Greedy matching: repeatedly pick smallest distance
            while True:
                idx = np.unravel_index(np.argmin(distances), distances.shape)
                minval = distances[idx]
                if np.isinf(minval):
                    break
                t_idx, d_idx = idx
                if minval < MAX_MATCH_DIST:
                    # assign
                    assigned_tracks.add(t_idx)
                    assigned_dets.add(d_idx)
                    # update track
                    self.tracks[t_idx].centroid = detections[d_idx]['centroid']
                    self.tracks[t_idx].keypoints = detections[d_idx]['keypoints']
                    self.tracks[t_idx].disappeared = 0
                    self.tracks[t_idx].history.append(detections[d_idx]['centroid'])
                    # set row/col to inf so they are not used again
                    distances[t_idx, :] = np.inf
                    distances[:, d_idx] = np.inf
                else:
                    break

        # Tracks not assigned -> mark disappeared
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.disappeared += 1

        # Detections not assigned -> create new tracks
        for j, d in enumerate(detections):
            if j not in assigned_dets:
                newt = Track(self.next_id, d['centroid'], d['keypoints'])
                self.next_id += 1
                self.tracks.append(newt)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.disappeared <= MAX_DISAPPEARED_FRAMES]
        return self.tracks

# ---------- POSE ESTIMATION UTILITIES ----------
def get_keypoints_from_net_output(output, frame_width, frame_height, thresh=KEYPOINT_THRESHOLD):
    """
    Parse output probability maps to extract keypoints for multiple people.
    This function is adapted/simplified from OpenCV OpenPose tutorial.
    It returns keypoints per detected person by using peaks in heatmaps and connecting via PAFs.
    NOTE: The OpenCV DNN model outputs parts and PAFs — full multi-person reconstruction can get lengthy.
    Here we use a simple approach: find peaks in each part heatmap, then group them by proximity to form persons.
    This won't be perfect for extremely crowded scenes, but works well for normal multi-person webcam scenes.
    """
    # output shape: [1, N, H, W], where first N parts are keypoint heatmaps, remaining are PAFs
    H = output.shape[2]
    W = output.shape[3]
    points_all = []  # for each part, list of found points: (x, y, prob)
    for part in range(N_POINTS):
        prob_map = output[0, part, :, :]
        prob_map = cv2.resize(prob_map, (frame_width, frame_height))
        # find peaks
        map_smooth = cv2.GaussianBlur(prob_map, (3,3), 0, 0)
        _, prob_thresh, _, max_loc = cv2.minMaxLoc(map_smooth)
        keypoints = []
        # simple peak extraction: threshold and non-max suppression via dilation
        map_mask = np.where(map_smooth > thresh, 1, 0).astype(np.uint8)
        # find contours around mask regions
        contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            blob_mask = np.zeros_like(map_smooth, dtype=np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 1, -1)
            masked = map_smooth * blob_mask
            _, maxVal, _, maxPoint = cv2.minMaxLoc(masked)
            if maxVal > thresh:
                kp = (int(maxPoint[0]), int(maxPoint[1]), float(maxVal))
                keypoints.append(kp)
        points_all.append(keypoints)

    # Now we have candidate points for each part; group them to persons by simple greedy merging:
    # Start with person candidates from detected '1' (neck/chest) or '0' (nose) - fallback to all points.
    # We'll create person groups by proximity of detected points.
    persons = []
    # Gather all keypoints as flat list with (part_idx, x,y,prob)
    flat = []
    for part_idx, kps in enumerate(points_all):
        for (x,y,prob) in kps:
            flat.append({'part': part_idx, 'x': x, 'y': y, 'prob': prob})
    # Greedy grouping: seed a person with an unassigned keypoint, then attach nearest keypoints from other parts if within distance
    assigned = [False]*len(flat)
    for i, seed in enumerate(flat):
        if assigned[i]:
            continue
        person_kps = [None]*N_POINTS
        person_kps[seed['part']] = (seed['x'], seed['y'], seed['prob'])
        assigned[i] = True
        # look for other points to attach
        for j, cand in enumerate(flat):
            if assigned[j]:
                continue
            dist = np.hypot(seed['x']-cand['x'], seed['y']-cand['y'])
            # tolerance grows a bit with image size
            if dist < 100:
                if person_kps[cand['part']] is None:
                    person_kps[cand['part']] = (cand['x'], cand['y'], cand['prob'])
                    assigned[j] = True
        # compute centroid of assigned points
        xs = [p[0] for p in person_kps if p is not None]
        ys = [p[1] for p in person_kps if p is not None]
        if len(xs) >= 1:
            centroid = (int(sum(xs)/len(xs)), int(sum(ys)/len(ys)))
            persons.append({'keypoints': person_kps, 'centroid': centroid})
    return persons

# ---------- MAIN ----------
def main():
    print("Loading network...")
    net = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)
    # if you have a GPU-enabled OpenCV build:
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    cap = cv2.VideoCapture(0)  # change index if multiple cameras
    if not cap.isOpened():
        print("Cannot open camera. Exiting.")
        return

    tracker = SimpleTracker()
    fps_time = 0

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        frame_height, frame_width = frame.shape[:2]

        inp_blob = cv2.dnn.blobFromImage(frame, 1.0/255, (IN_WIDTH, IN_HEIGHT),
                                         (0,0,0), swapRB=False, crop=False)
        net.setInput(inp_blob)
        output = net.forward()  # shape [1, N, H, W]

        # parse multi-person keypoints (simplified approach)
        persons = get_keypoints_from_net_output(output, frame_width, frame_height)

        # prepare detections for tracker
        detections = []
        for p in persons:
            detections.append({'centroid': p['centroid'], 'keypoints': p['keypoints']})

        tracks = tracker.update(detections)

        # Draw
        vis = frame.copy()
        # draw each person's keypoints and skeleton
        for t in tracks:
            kp_list = t.keypoints
            # draw keypoints
            for idx, kp in enumerate(kp_list):
                if kp is None: continue
                x,y,prob = kp
                cv2.circle(vis, (int(x), int(y)), 4, (0,255,0), -1)
                cv2.putText(vis, str(idx), (int(x)+4, int(y)+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            # draw skeleton lines
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]
                if kp_list[partA] is None or kp_list[partB] is None:
                    continue
                x1,y1,_ = kp_list[partA]
                x2,y2,_ = kp_list[partB]
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

            # draw ID and centroid
            cx, cy = t.centroid
            cv2.putText(vis, f"ID {t.id}", (int(cx)-10, int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.circle(vis, (int(cx), int(cy)), 6, (0,0,255), -1)

            # optional: draw history trail
            for i in range(1, len(t.history)):
                cv2.line(vis, t.history[i-1], t.history[i], (0, 255, 255), 2)

        # FPS
        ctime = time.time()
        fps = 1 / (ctime - fps_time) if fps_time != 0 else 0
        fps_time = ctime
        cv2.putText(vis, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Multi-person Pose Tracker", vis)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
