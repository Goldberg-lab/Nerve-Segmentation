import torch
torch.backends.mps.is_available = lambda: False

import streamlit as st
import numpy as np
import cv2
from inference import get_model
import supervision as sv
import os
import io
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
try:
    # Fallback click-capture component that reliably renders images on Streamlit Cloud
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None
from pathlib import Path
import zipfile
import shutil


ROBOFLOW_API_KEY = (
    os.environ.get("ROBOFLOW_API_KEY")
    or st.secrets.get("ROBOFLOW_API_KEY", "")
)
# Compatible rerun helper for old/new Streamlit
def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# --- CHANGE THIS SECTION ---
@st.cache_resource  # Clean, modern caching for ML models
def load_model():
    key = ROBOFLOW_API_KEY or os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        st.error("Missing ROBOFLOW_API_KEY.")
        st.stop()
    return get_model(model_id="mouse-optic-nerve-uktj7/12", api_key=key)

model = load_model()


if "app_step" not in st.session_state:
    st.session_state.app_step = "upload"
if "yellow_mask" not in st.session_state:
    st.session_state.yellow_mask = None
if "orig_shape" not in st.session_state:
    st.session_state.orig_shape = None
if "rightmost_point" not in st.session_state:
    st.session_state.rightmost_point = None
if "top_leftmost_point" not in st.session_state:
    st.session_state.top_leftmost_point = None
if "bottom_leftmost_point" not in st.session_state:
    st.session_state.bottom_leftmost_point = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_img_idx" not in st.session_state:
    st.session_state.current_img_idx = 0
if "csv_buffers" not in st.session_state:
    st.session_state.csv_buffers = {}
if "microns_per_pixel" not in st.session_state:
    st.session_state["microns_per_pixel"] = 3.07
if "interval_microns" not in st.session_state:
    # default interval = current sampling radius (30 px) * microns_per_pixel
    st.session_state["interval_microns"] = 30 * st.session_state.get("microns_per_pixel", 3.07)
if "sampling_radius" not in st.session_state:
    st.session_state["sampling_radius"] = 30


st.title("Optic Nerve Mask Segmentation")


if st.session_state.app_step == "upload":
    st.write("Welcome to the Optic Nerve Mask Segmentation App! This app allows you to upload a folder of optic nerve images, run inference to segment each nerve, and then select chiasm points for further analysis.")

    # microns-per-pixel control shown only on upload step (writes into session_state)
    st.number_input(
        "Image Scale: Microns per pixel (µm/pixel)",
        min_value=0.0001,
        step=0.01,
        format="%.4f",
        key="microns_per_pixel",
        help="Enter the number of microns represented by one pixel for your imaging setup. Default: 3.07"
    )

    # measurement interval in microns (used to compute sampling radius in pixels)
    st.number_input(
        "Measurement interval: microns between sampling lines (µm)",
        min_value=0.01,
        step=0.1,
        format="%.2f",
        key="interval_microns",
        help="Distance between consecutive measurement sections in microns. Default = 30 px × µm/pixel"
    )

    uploaded_files = st.file_uploader(
        "Upload a folder of optic nerve images", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected.")
        # require explicit confirmation so widget value is saved before changing app_step
        # ...existing code...
        if st.button("➡️ Start processing"):
            # copy current widget value into a separate confirmed key (safe to write)
            st.session_state["microns_per_pixel_confirmed"] = float(
                st.session_state.get("microns_per_pixel", 3.07)
            )
            # confirm interval and compute sampling radius (px)
            st.session_state["interval_microns_confirmed"] = float(
                st.session_state.get("interval_microns", 30 * st.session_state.get("microns_per_pixel", 3.07))
            )
            # sampling radius in pixels (rounded int) used in contour stepping
            sampling_px = st.session_state["interval_microns_confirmed"] / st.session_state["microns_per_pixel_confirmed"]
            st.session_state["sampling_radius"] = max(1, int(round(sampling_px)))
            st.session_state.uploaded_files = uploaded_files
            st.session_state.current_img_idx = 0
            st.session_state.csv_buffers = {}
            st.session_state.app_step = "model"
            do_rerun()
# ...existing code...

if st.session_state.app_step == "model":
    uploaded_files = st.session_state.uploaded_files
    current_idx = st.session_state.current_img_idx
    uploaded_file = uploaded_files[current_idx]
    filename_base = os.path.splitext(uploaded_file.name)[0]
    st.session_state.uploaded_filename = filename_base
    st.write(f"**Image {current_idx + 1} of {len(uploaded_files)}:** `{uploaded_file.name}`")

    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Original Image", channels="BGR")

    # Save original dimensions
    st.session_state.orig_shape = image.shape[:2]  # (H, W)

    # Run inference
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)
    masks = detections.mask  # shape: (N, H, W)

    if len(masks) == 0:
        st.warning("⚠️ No masks found.")
        st.stop()
    else:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        if len(masks) >= 2:
            areas = [np.sum(mask) for mask in masks]
            idx_outer = np.argmax(areas)
            idx_inner = np.argmin(areas)

            mask_outer = (masks[idx_outer].astype(np.uint8)) * 255
            mask_inner = (masks[idx_inner].astype(np.uint8)) * 255
            yellow_mask = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

            axs[1].imshow(mask_outer, cmap="gray")
            axs[1].set_title("Outer Mask")
            axs[1].axis("off")

            axs[2].imshow(yellow_mask, cmap="gray")
            axs[2].set_title("Refined (Outer - Inner)")
            axs[2].axis("off")
        else:
            yellow_mask = (masks[0].astype(np.uint8)) * 255
            axs[1].imshow(yellow_mask, cmap="gray")
            axs[1].set_title("Refined Mask (Single)")
            axs[1].axis("off")
            axs[2].axis("off")

        st.pyplot(fig)

        st.session_state.yellow_mask = yellow_mask

    if st.button("➡️ Next: Select Points"):
        st.session_state.app_step = "select"
        do_rerun()

# --- STEP 2: POINT SELECTION ---
if st.session_state.app_step == "select":

    uploaded_files = st.session_state.uploaded_files
    current_idx = st.session_state.current_img_idx
    uploaded_file = uploaded_files[current_idx]
    filename_base = os.path.splitext(uploaded_file.name)[0]
    st.session_state.uploaded_filename = filename_base
    st.write(f"**Image {current_idx + 1} of {len(uploaded_files)}:** `{uploaded_file.name}`")

    st.subheader("Select Chiasm & Leg Endpoints")
    st.markdown("Click **three** points on the nerve:")
    st.markdown("1. The rightmost point (chiasm)\n2. The end of the **top** leg\n3. The end of the **bottom** leg\n*(Order doesn't matter, the app will sort them automatically!)*")

    yellow_mask = st.session_state.yellow_mask
    orig_h, orig_w = st.session_state.orig_shape

    display_width = 600
    scale_factor = display_width / orig_w
    display_height = int(orig_h * scale_factor)

    current_image_key = f"{current_idx}_{filename_base}"
    display_cache_key = f"_display_pil_bg_{current_image_key}_{display_width}"

    if st.session_state.get(display_cache_key) is None:
        resized = cv2.resize(yellow_mask, (display_width, display_height))
        display_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        st.session_state[display_cache_key] = Image.fromarray(display_img).convert("RGB")

    pil_bg = st.session_state[display_cache_key]

    if "clicked_points_display" not in st.session_state:
        st.session_state.clicked_points_display = []

    if st.session_state.get("_last_select_image_key") != current_image_key:
        st.session_state.clicked_points_display = []
        st.session_state.rightmost_point = None
        st.session_state.top_leftmost_point = None
        st.session_state.bottom_leftmost_point = None
        st.session_state._last_select_image_key = current_image_key

    if st.button("Reset selected points"):
        st.session_state.clicked_points_display = []
        st.session_state.rightmost_point = None
        st.session_state.top_leftmost_point = None
        st.session_state.bottom_leftmost_point = None
        do_rerun()

    if streamlit_image_coordinates is None:
        st.error("`streamlit-image-coordinates` is not installed.")
    else:
        preview = pil_bg.copy()
        draw = ImageDraw.Draw(preview)
        r = 6
        for (px, py) in st.session_state.clicked_points_display:
            draw.ellipse((px - r, py - r, px + r, py + r), outline=(0, 255, 255), width=3)

        click = streamlit_image_coordinates(preview, key=f"img_click_{current_idx}_{filename_base}")
        if click is not None and "x" in click and "y" in click:
            x_disp = int(click["x"])
            y_disp = int(click["y"])

            # Collect up to 3 distinct clicks
            if len(st.session_state.clicked_points_display) < 3:
                is_new = True
                for (px, py) in st.session_state.clicked_points_display:
                    if abs(x_disp - px) <= 2 and abs(y_disp - py) <= 2:
                        is_new = False
                if is_new:
                    st.session_state.clicked_points_display.append((x_disp, y_disp))
                    do_rerun()

        # Once 3 points are collected, process them
        if len(st.session_state.clicked_points_display) >= 3:
            p1, p2, p3 = st.session_state.clicked_points_display[:3]

            # Map display -> original high-res coords
            pts = [
                (int(round(x / scale_factor)), int(round(y / scale_factor)))
                for x, y in (p1, p2, p3)
            ]

            # 1. Rightmost point has the largest X
            pts.sort(key=lambda pt: pt[0], reverse=True)
            rightmost = pts[0]

            # 2. The remaining two are the left legs. Sort by Y (smaller Y = Top)
            left_pts = pts[1:]
            left_pts.sort(key=lambda pt: pt[1])
            top_leftmost = left_pts[0]
            bottom_leftmost = left_pts[1]

            st.session_state.rightmost_point = rightmost
            st.session_state.top_leftmost_point = top_leftmost
            st.session_state.bottom_leftmost_point = bottom_leftmost

            st.success(f"✅ Rightmost X: {rightmost[0]} | Top Leg X: {top_leftmost[0]} | Bottom Leg X: {bottom_leftmost[0]}")

            if st.button("➡️ Next: View Diameter Visualization and Graph"):
                st.session_state.app_step = "diameter"
                do_rerun()
        else:
            st.info(f"ℹ️ Click {3 - len(st.session_state.clicked_points_display)} more point(s) on the image.")



# --- STEP 3: DIAMETER MEASUREMENT ---

if st.session_state.app_step == "diameter":
    
    
    

    uploaded_files = st.session_state.uploaded_files
    current_idx = st.session_state.current_img_idx
    uploaded_file = uploaded_files[current_idx]
    filename_base = os.path.splitext(uploaded_file.name)[0]
    st.session_state.uploaded_filename = filename_base
    st.write(f"**Image {current_idx + 1} of {len(uploaded_files)}:** `{uploaded_file.name}`")

    st.subheader("Nerve Diameter Measurement")

    # analyze ------------------------

    rightmost_x = st.session_state.rightmost_point
    top_leftmost_x = st.session_state.top_leftmost_point
    bottom_leftmost_x = st.session_state.bottom_leftmost_point

    
    yellow_mask = st.session_state.yellow_mask
    orig_h, orig_w = st.session_state.orig_shape

    radius = int(st.session_state.get("sampling_radius", 30))
    angle_step = 1
    max_steps = 100

    kernel = np.ones((3, 3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    _, yellow_mask = cv2.threshold(yellow_mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        st.error("❌ No contours found.")
        st.stop()

    main_contour = max(contours, key=cv2.contourArea)
    contour_points = set(tuple(pt[0]) for pt in main_contour)

    rx = rightmost_x[0]
    lx_top = top_leftmost_x[0]       # Top leg stopping point
    lx_bottom = bottom_leftmost_x[0] # Bottom leg stopping point
    column = yellow_mask[:, rx]
    nonzero_y = np.where(column > 0)[0]
    if len(nonzero_y) < 2:
        st.error("Could not find top and bottom at rightmost_x")
        st.stop()

    top_start = (rx, nonzero_y[0])
    bottom_start = (rx, nonzero_y[-1])
    top_path = [top_start]
    bottom_path = [bottom_start]

    def find_next_contour_point(cx, cy, radius, min_angle, max_angle):
        angles = range(min_angle, max_angle + 1, angle_step) if min_angle <= max_angle else range(min_angle, max_angle - 1, -angle_step)
        best_pt = None
        min_dist = float('inf')

        for angle in angles:
            rad = np.deg2rad(angle)
            x = int(round(cx + radius * np.cos(rad)))
            y = int(round(cy + radius * np.sin(rad)))
            
            # Check distance to main outer contour
            dist = abs(cv2.pointPolygonTest(main_contour, (x, y), True))
            if dist < 8:  # Relaxed tolerance so small curvature steps aren't dropped
                if dist < min_dist:
                    min_dist = dist
                    best_pt = (x, y)

        return best_pt

    # Walk the top path (towards the left: angles pointing generally West/Northwest/Southwest)
    for _ in range(max_steps):
        cx, cy = top_path[-1]
        # 100 deg (down-left) to 260 deg (up-left), centered around 180 deg (straight left)
        next_pt = find_next_contour_point(cx, cy, radius, 260, 100)
        if not next_pt or next_pt[0] < lx_top:
            break
        top_path.append(next_pt)

    # Walk the bottom path (towards the left)
    for _ in range(max_steps):
        cx, cy = bottom_path[-1]
        # 80 deg to 280 deg centered around 180 deg
        next_pt = find_next_contour_point(cx, cy, radius, 100, 260)
        if not next_pt or next_pt[0] < lx_bottom:
            break
        bottom_path.append(next_pt)

    # === START: Midpoint Intersection Analysis with Bridge Separator ===
    _, binary_mask = cv2.threshold(yellow_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    visualization = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)

    top_x_values = [pt[0] for pt in top_path]
    bottom_x_values = [pt[0] for pt in bottom_path]
    red_lines = sorted(list(set(top_x_values[1:] + bottom_x_values)))

    raw_midpoints = {}      # x -> y midpoint
    is_touching = {}        # x -> True if legs are fused

    for x in red_lines:
        intersections = []
        for contour in contours:
            for i in range(len(contour) - 1):
                pt1, pt2 = contour[i][0], contour[i + 1][0]
                if pt1[0] <= x <= pt2[0] or pt2[0] <= x <= pt1[0]:
                    intersections.append((pt1[1], pt2[1]))

        if not intersections:
            continue

        intersections = sorted(intersections, key=lambda t: min(t[0], t[1]))
        groups = []
        current_group = [intersections[0]]

        for i in range(1, len(intersections)):
            y1, y2 = intersections[i]
            last_y1, last_y2 = current_group[-1]
            if abs(y1 - last_y1) <= 3 or abs(y2 - last_y2) <= 3:
                current_group.append((y1, y2))
            else:
                groups.append(current_group)
                current_group = [(y1, y2)]
        groups.append(current_group)

        if len(groups) >= 4:
            # Clean split: top nerve, space, bottom nerve
            y2_avg = np.mean([y for y, _ in groups[1]])
            y3_avg = np.mean([y for y, _ in groups[2]])
            raw_midpoints[x] = int((y2_avg + y3_avg) / 2)
            is_touching[x] = False
        elif len(groups) == 3:
            # Touching at single boundary line
            y_mid = np.mean([y for y, _ in groups[1]])
            raw_midpoints[x] = int(y_mid)
            is_touching[x] = False
        else:
            # 1 or 2 groups: Fused / touching zone
            is_touching[x] = True
            raw_midpoints[x] = None

    # Bridge separator interpolation across touching zones
    final_midpoints_dict = dict(raw_midpoints)
    x_keys = sorted(red_lines)

    for idx, x in enumerate(x_keys):
        if is_touching.get(x, False):
            # 1. Search left for the nearest separated anchor
            left_anchor = None
            for l_idx in range(idx - 1, -1, -1):
                lx = x_keys[l_idx]
                if not is_touching.get(lx, True) and raw_midpoints[lx] is not None:
                    left_anchor = (lx, raw_midpoints[lx])
                    break

            # 2. Search right for the nearest separated anchor
            right_anchor = None
            for r_idx in range(idx + 1, len(x_keys)):
                rx_val = x_keys[r_idx]
                if not is_touching.get(rx_val, True) and raw_midpoints[rx_val] is not None:
                    right_anchor = (rx_val, raw_midpoints[rx_val])
                    break

            # 3. Connect anchors with linear separator line
            if left_anchor and right_anchor:
                x0, y0 = left_anchor
                x1, y1 = right_anchor
                interpolated_y = y0 + (x - x0) * (y1 - y0) / float(x1 - x0)
                final_midpoints_dict[x] = int(round(interpolated_y))
            elif left_anchor:
                final_midpoints_dict[x] = left_anchor[1]
            elif right_anchor:
                final_midpoints_dict[x] = right_anchor[1]
            else:
                final_midpoints_dict[x] = orig_h // 2

    # Map back into the list format expected by the downstream code
    midpoints = [(x, final_midpoints_dict[x]) for x in red_lines if x in final_midpoints_dict]

    # Draw the separator line on the preview
    if len(midpoints) > 1:
        for i in range(1, len(midpoints)):
            pt1 = midpoints[i - 1]
            pt2 = midpoints[i]
            # Draw yellow separator seam
            cv2.line(visualization, pt1, pt2, (0, 255, 255), 2)
            # Highlight touching / bridged points in magenta
            if is_touching.get(pt2[0], False):
                cv2.circle(visualization, pt2, 4, (255, 0, 255), -1)

    st.session_state.yellow_mask_processed = yellow_mask
    st.session_state.top_x_values = top_x_values
    st.session_state.bottom_x_values = bottom_x_values
    st.session_state.midpoints = midpoints
    st.session_state.midpoints_dict = final_midpoints_dict  





    # analyze ---------------------------







    # Retrieve values from session state
    yellow_mask = st.session_state.get("yellow_mask_processed")
    top_x_values = st.session_state.get("top_x_values")
    bottom_x_values = st.session_state.get("bottom_x_values")
    midpoints = st.session_state.get("midpoints")

    

    if yellow_mask is None or top_x_values is None or bottom_x_values is None or midpoints is None:
        st.error("Required data for diameter measurement is missing. Please run contour analysis first.")
        st.stop()

    # Convert yellow_mask to grayscale if needed
    if len(yellow_mask.shape) == 3:
        yellow_mask = cv2.cvtColor(yellow_mask, cv2.COLOR_BGR2GRAY)

    color_mask = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
    height, width = yellow_mask.shape
    np.random.seed(42)

    # Storage for top and bottom points with section index
    top_points = []  # Will store (x, y, section_index)
    bottom_points = []  # Will store (x, y, section_index)

    color = (0, 255, 255)  # yellow for points

    # --- Top points directly from path ---
    for i, (x, y) in enumerate(top_path):
        top_points.append((x, y, i))
        cv2.circle(color_mask, (x, y), 6, color, -1)

    # --- Bottom points directly from path ---
    for i, (x, y) in enumerate(bottom_path):
        bottom_points.append((x, y, i))
        cv2.circle(color_mask, (x, y), 6, color, -1)

    # Function to estimate tangent slope using nearby points
    def estimate_tangent_slope(points, index, search_range=3):
        """Estimate the tangent slope at a given index using nearby points."""
        if len(points) < 2:
            return None  # Not enough points to estimate slope

        x, y, _ = points[index]

        # Find two nearby points for slope estimation
        left_idx = max(0, index - search_range)
        right_idx = min(len(points) - 1, index + search_range)

        x1, y1, _ = points[left_idx]
        x2, y2, _ = points[right_idx]

        if x2 - x1 == 0:
            return None  # Avoid division by zero

        return (y2 - y1) / (x2 - x1)  # Slope = rise / run

    def segments_intersect(p1, p2, p3, p4):
        """Standard segment-intersection test (proper crossing, not just touching)."""
        def ccw(a, b, c):
            return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

    # Function to find intersection with mask boundary along a line with midpoint constraint
    def find_mask_intersection(mask, start_x, start_y, angle, midpoint_y, is_top, max_length=300):
        """Find the intersection point with mask boundary starting from (start_x, start_y)
        and moving in the direction given by angle (in radians).

        Parameters:
        - mask: The binary mask image
        - start_x, start_y: Starting point coordinates
        - angle: Direction angle in radians
        - midpoint_y: Y-coordinate of midpoint line (for stopping condition)
        - is_top: Whether this is for the top nerve (True) or bottom nerve (False)
        - max_length: Maximum ray length to check
        """
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Check points along the ray
        for length in range(1, max_length):
            x = int(start_x + length * dx)
            y = int(start_y + length * dy)

            # Check if out of bounds
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                return None

            # Check if we've reached the midpoint boundary
            # For top nerve, stop if we go below midpoint (when going down)
            if is_top and dy > 0 and y >= midpoint_y:
                return (x, y)

            # For bottom nerve, stop if we go above midpoint (when going up)
            if not is_top and dy < 0 and y <= midpoint_y:
                return (x, y)

            # Check if we've hit the boundary (pixel value changes from >0 to 0)
            if mask[y, x] == 0:
                return (x, y)

        return None  # No intersection found within max_length

    def vertical_diameter_top_leg(mask, start_x, start_y, midpoint_y, go_down=True, max_length=300):
        """
        For index == 1: Draw a vertical line up or down from (start_x, start_y),
        stopping when reaching midpoint_y or exiting mask.
        """
        direction = 1 if go_down else -1

        for length in range(1, max_length):
            x = start_x
            y = start_y + length * direction  # Move vertically

            # Out of bounds
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                return None

            # If reached midpoint_y, stop
            if go_down and y >= midpoint_y:
                return (x, y)
            if not go_down and y <= midpoint_y:
                return (x, y)

            # If pixel is outside mask, stop
            if mask[y, x] == 0:
                return (x, y)

        return None

    def vertical_diameter_bottom_leg(mask, start_x, start_y, midpoint_y, go_up=True, max_length=300):
        """
        For bottom leg: Draw a vertical line up or down from (start_x, start_y),
        stopping when reaching midpoint_y or exiting mask.
        """
        direction = -1 if go_up else 1

        for length in range(1, max_length):
            x = start_x
            y = start_y + length * direction  # Move vertically

            # Out of bounds
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                return None

            # If reached midpoint_y, stop
            if go_up and y <= midpoint_y:
                return (x, y)
            if not go_up and y >= midpoint_y:
                return (x, y)

            # If pixel is outside mask, stop
            if mask[y, x] == 0:
                return (x, y)

        return None


    def cast_ray_to_target(mask, start_x, start_y, target_x, target_y, midpoint_y, is_top, max_length=400):
        """
        Casts a ray from (start_x, start_y) directed towards (target_x, target_y)
        and stops at the mask edge or the midpoint boundary.
        """
        dx = target_x - start_x
        dy = target_y - start_y
        dist = np.hypot(dx, dy)
        if dist == 0:
            return (start_x, start_y)
        
        ux, uy = dx / dist, dy / dist
        last_valid = (start_x, start_y)

        for step in range(1, max_length):
            cx = int(round(start_x + step * ux))
            cy = int(round(start_y + step * uy))

            # Check bounds
            if cx < 0 or cx >= mask.shape[1] or cy < 0 or cy >= mask.shape[0]:
                break

            # Stop at midpoint divider
            if is_top and cy >= midpoint_y:
                return (cx, cy)
            if not is_top and cy <= midpoint_y:
                return (cx, cy)

            # Stop if leaving the mask
            if mask[cy, cx] == 0:
                return (cx, cy)

            last_valid = (cx, cy)

        return last_valid

        
    def compute_smoothed_diameters(points, mask, midpoints_dict, height, is_top=True):
        """
        Computes diameter segments:
        - Index 0: strictly vertical
        - Interpolates intermediate lines that cross index 0
        - Index K+: normal perpendicular lines
        """
        if not points:
            return []

        # 1. Compute candidate segments
        candidates = []  # list of tuples: (outer_pt, inner_pt, valid_perp)

        for i, (x, y, section_idx) in enumerate(points):
            midpoint_y = midpoints_dict.get(x, height // 2)
            outer_pt = (x, y)

            if i == 0:
                # First line is strictly vertical
                if is_top:
                    in2 = vertical_diameter_top_leg(mask, x, y, midpoint_y, go_down=True)
                else:
                    in2 = vertical_diameter_bottom_leg(mask, x, y, midpoint_y, go_up=True)
                inner_pt = in2 if in2 else (x, midpoint_y)
                candidates.append((outer_pt, inner_pt, False))
                continue

            # Perpendicular candidates
            slope = estimate_tangent_slope(points, i)
            if slope is None or slope == 0:
                perp_angle = np.pi / 2
            else:
                perp_angle = np.arctan(-1 / slope)

            # Ensure the ray always points inward toward the middle of the nerve
            # For top nerve: inward means dy > 0 (pointing down)
            # For bottom nerve: inward means dy < 0 (pointing up)
            dy = np.sin(perp_angle)
            if (is_top and dy < 0) or (not is_top and dy > 0):
                perp_angle += np.pi

            in_pt = find_mask_intersection(mask, x, y, perp_angle, midpoint_y, is_top)

            if in_pt is None:
                # Fallback if ray misses
                in_pt = (x, midpoint_y)

            candidates.append((outer_pt, in_pt, True))

        # 2. Find the first perpendicular segment (K) that does NOT cross the index 0 vertical segment
        p0_outer, p0_inner, _ = candidates[0]
        first_clean_idx = len(candidates) - 1  # default if all intersect

        for i in range(1, len(candidates)):
            pi_outer, pi_inner, _ = candidates[i]
            # Check intersection with index 0
            if not segments_intersect(p0_outer, p0_inner, pi_outer, pi_inner):
                first_clean_idx = i
                break

        # 3. Build smoothed list
        final_segments = []
        # Index 0
        final_segments.append((p0_outer, p0_inner))

        # Interpolate segments 1 to first_clean_idx - 1
        if first_clean_idx > 1:
            clean_inner_x = candidates[first_clean_idx][1][0]
            base_inner_x = p0_inner[0]

            for i in range(1, first_clean_idx):
                outer_x, outer_y = candidates[i][0]
                mid_y = midpoints_dict.get(outer_x, height // 2)

                alpha = i / float(first_clean_idx)
                target_x = int(round(base_inner_x + alpha * (clean_inner_x - base_inner_x)))
                target_y = mid_y

                smooth_inner = cast_ray_to_target(mask, outer_x, outer_y, target_x, target_y, mid_y, is_top)

                # IF THE RAY COLLAPSED (diameter < 5), PREVENT CROSSING BY BORROWING NEIGHBOR RAY DIRECTION
                if np.hypot(smooth_inner[0] - outer_x, smooth_inner[1] - outer_y) < 5:
                    # Look at the previous line's vector
                    prev_out, prev_in = final_segments[-1]
                    dx = prev_in[0] - prev_out[0]
                    dy = prev_in[1] - prev_out[1]

                    # Aim along that same direction from the current outer point
                    smooth_inner = cast_ray_to_target(
                        mask, 
                        outer_x, 
                        outer_y, 
                        outer_x + dx, 
                        outer_y + dy, 
                        mid_y, 
                        is_top
                    )

                final_segments.append(((outer_x, outer_y), smooth_inner))

        # Append normal perpendicular lines from first_clean_idx onward
        for i in range(first_clean_idx, len(candidates)):
            if i == 0:
                continue
            outer_pt, inner_pt, _ = candidates[i]
            
            # Check against previous segment to prevent consecutive overlaps
            prev_outer, prev_inner = final_segments[-1]
            if segments_intersect(outer_pt, inner_pt, prev_outer, prev_inner):
                mid_y = midpoints_dict.get(outer_pt[0], height // 2)
                target_x = outer_pt[0] + (prev_inner[0] - prev_outer[0])
                inner_pt = cast_ray_to_target(mask, outer_pt[0], outer_pt[1], target_x, mid_y, mid_y, is_top)

            final_segments.append((outer_pt, inner_pt))

        return final_segments

    # Lists to store diameter measurements
    diameters_top = []
    diameters_bottom = []
    midpoints_dict = st.session_state.get("midpoints_dict", dict(midpoints))

    # Process Top Leg
    top_segments = compute_smoothed_diameters(top_points, yellow_mask, midpoints_dict, height, is_top=True)
    for (x1, y1), (x2, y2) in top_segments:
        diameter = np.hypot(x2 - x1, y2 - y1)
        diameters_top.append((x1, diameter))
        cv2.line(color_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Process Bottom Leg
    bottom_segments = compute_smoothed_diameters(bottom_points, yellow_mask, midpoints_dict, height, is_top=False)
    for (x1, y1), (x2, y2) in bottom_segments:
        diameter = np.hypot(x2 - x1, y2 - y1)
        diameters_bottom.append((x1, diameter))
        cv2.line(color_mask, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw the midpoint dots to visualize the midpoints
    for i, midpoint in enumerate(midpoints):
        cv2.circle(color_mask, (int(midpoint[0]), int(midpoint[1])), 4, (255, 255, 0), -1)

    # Display the result with measurements in Streamlit
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    ax1.set_xlim(0, width)
    ax1.set_ylim(height, 0)
    ax1.set_xlabel('Position Value (How Far Along the Nerve)')
    ax1.set_ylabel('Y-axis')
    ax1.set_title('Nerve Diameter Measurements Graph')
    st.pyplot(fig1)

      # Plot diameter vs x position in Streamlit
    radius = int(st.session_state.get("sampling_radius", 30))
    MICRONS_PER_PIXEL = float(
        st.session_state.get(
            "microns_per_pixel_confirmed",
            st.session_state.get("microns_per_pixel", 3.07),
        )
    )

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if diameters_top:
        positions_top_px = [i * radius for i in range(len(diameters_top))]
        d_values_top_px = [d for x, d in diameters_top]
        # Convert to microns
        positions_top_um = [p * MICRONS_PER_PIXEL for p in positions_top_px]
        d_values_top_um = [d * MICRONS_PER_PIXEL for d in d_values_top_px]
        ax2.plot(positions_top_um, d_values_top_um, 'go-', label='Top Nerve')

    if diameters_bottom:
        positions_bottom_px = [i * radius for i in range(len(diameters_bottom))]
        d_values_bottom_px = [d for x, d in diameters_bottom]
        # Convert to microns
        positions_bottom_um = [p * MICRONS_PER_PIXEL for p in positions_bottom_px]
        d_values_bottom_um = [d * MICRONS_PER_PIXEL for d in d_values_bottom_px]
        ax2.plot(positions_bottom_um, d_values_bottom_um, 'ro-', label='Bottom Nerve')

    ax2.set_xlabel('Position along Nerve (microns)')
    ax2.set_ylabel('Diameter (microns)')
    ax2.set_title('Nerve Diameter vs Position')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # --- CSV Download Option ---
    import pandas as pd
    from io import StringIO
    
     # Prepare data for CSV: combine top and bottom nerves side by side
    max_len = max(len(diameters_top), len(diameters_bottom))
    MICRONS_PER_PIXEL = float(
        st.session_state.get(
            "microns_per_pixel_confirmed",
            st.session_state.get("microns_per_pixel", 3.07),
        )
    )

    rows = []
    for i in range(max_len):
        # Top nerve data
        if i < len(diameters_top):
            x_top, d_top = diameters_top[i]
            pos_top_px = i * radius
            pos_top_um = pos_top_px * MICRONS_PER_PIXEL
            d_top_um = d_top * MICRONS_PER_PIXEL
        else:
            x_top, d_top, pos_top_px, pos_top_um, d_top_um = [None]*5

        # Bottom nerve data
        if i < len(diameters_bottom):
            x_bot, d_bot = diameters_bottom[i]
            pos_bot_px = i * radius
            pos_bot_um = pos_bot_px * MICRONS_PER_PIXEL
            d_bot_um = d_bot * MICRONS_PER_PIXEL
        else:
            x_bot, d_bot, pos_bot_px, pos_bot_um, d_bot_um = [None]*5

        rows.append({
            "Top_X": x_top,
            "Top_Position_along_nerve_px": pos_top_px,
            "Top_Diameter_px": d_top,
            "Top_Position_along_nerve_um": pos_top_um,
            "Top_Diameter_um": d_top_um,
            "Bottom_X": x_bot,
            "Bottom_Position_along_nerve_px": pos_bot_px,
            "Bottom_Diameter_px": d_bot,
            "Bottom_Position_along_nerve_um": pos_bot_um,
            "Bottom_Diameter_um": d_bot_um
        })

    if rows:
        df = pd.DataFrame(rows)

        # Count how many columns are for top and bottom
        top_cols = [col for col in df.columns if col.startswith("Top_")]
        bottom_cols = [col for col in df.columns if col.startswith("Bottom_")]

        # Build the first header row: repeat "Top Nerve" for all top columns, "Bottom Nerve" for all bottom columns
        multi_header = ["Top Nerve"] * len(top_cols) + ["Bottom Nerve"] * len(bottom_cols)

        csv_buffer = StringIO()
        # Write the multi-header row
        csv_buffer.write(",".join(multi_header) + "\n")
        # Write the actual column headers and data
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Diameter Data as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{getattr(st.session_state, 'uploaded_filename', 'nerve')}_diameters.csv",
            mime="text/csv",
            key="download_diameter_csv"
        )
        csv_filename = f"{getattr(st.session_state, 'uploaded_filename', 'nerve')}_diameters.csv"
        st.session_state.csv_buffers[csv_filename] = StringIO(csv_buffer.getvalue())

        if st.session_state.current_img_idx < len(st.session_state.uploaded_files) - 1:
            if st.button("➡️ Next Image"):
                st.session_state.current_img_idx += 1
                st.session_state.app_step = "model"
                do_rerun()

        # Only show ZIP download on the last image
        import zipfile
        import io
        if st.session_state.current_img_idx == len(st.session_state.uploaded_files) - 1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for fname, csv_buf in st.session_state.csv_buffers.items():
                    zf.writestr(fname, csv_buf.getvalue())
            st.download_button(
                label="⬇️ Download All CSVs as ZIP",
                data=zip_buffer.getvalue(),
                file_name="all_nerve_diameters.zip",
                mime="application/zip"
            )
