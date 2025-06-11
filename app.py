import streamlit as st
import numpy as np
import cv2
from inference import get_model
import supervision as sv
import os
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Roboflow API Key
os.environ["ROBOFLOW_API_KEY"] = "rC3zob8rOUpOtd3W3bxY"

# Load model once
@st.cache_resource
def load_model():
    return get_model(model_id="mouse-optic-nerve-uktj7/6", api_key=os.environ["ROBOFLOW_API_KEY"])

model = load_model()

# Step controller
if "app_step" not in st.session_state:
    st.session_state.app_step = "upload"
if "yellow_mask" not in st.session_state:
    st.session_state.yellow_mask = None
if "orig_shape" not in st.session_state:
    st.session_state.orig_shape = None
if "rightmost_point" not in st.session_state:
    st.session_state.rightmost_point = None
if "leftmost_point" not in st.session_state:
    st.session_state.leftmost_point = None

st.title("🧠 Optic Nerve Mask Segmentation")


# --- STEP 1: UPLOAD + INFERENCE ---
if st.session_state.app_step == "upload":

    st.write("Welcome to the Optic Nerve Mask Segmentation App! This app allows you to upload an optic nerve image, run inference to segment the nerve, and then select chiasm points for further analysis.")
    uploaded_file = st.file_uploader("Upload an optic nerve image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Convert file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
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

            if st.button("➡️ Next: Select Chiasm Points"):
                st.session_state.app_step = "select"

# --- STEP 2: POINT SELECTION ---
if st.session_state.app_step == "select" and st.session_state.yellow_mask is not None:
    st.subheader("📍 Select Chiasm Points (Rightmost and Leftmost). The rightmost point should be a bit left to the chiasm, and the leftmost point should be about where the nerve ends.")
    st.markdown("👉 Click **first** on the rightmost chiasm point, then on the leftmost.")
    st.markdown("**Important:** Ensure that the leftmost point does not go beyond the edges of the optic nerve, and leave about 50 pixels of space on the edge of BOTH sides of the nerve in order to avoid errors.")


    yellow_mask = st.session_state.yellow_mask
    orig_h, orig_w = st.session_state.orig_shape

    display_width = 600
    scale_factor = display_width / orig_w
    display_height = int(orig_h * scale_factor)

    resized = cv2.resize(yellow_mask, (display_width, display_height))
    display_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(display_img)

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 0, 0.6)",
        stroke_color="cyan",
        stroke_width=3,
        background_image=pil_image,
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="point",
        point_display_radius=8,
        key="canvas_chiasm"
    )

    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) >= 2:
        coords = canvas_result.json_data["objects"]
        point1 = int(coords[0]["left"] / scale_factor), int(coords[0]["top"] / scale_factor)
        point2 = int(coords[1]["left"] / scale_factor), int(coords[1]["top"] / scale_factor)

        st.session_state.rightmost_point = point1
        st.session_state.leftmost_point = point2

        st.success(f"✅ Rightmost X: {point1[0]}, Leftmost X: {point2[0]}")

        if st.button("➡️ Next: Run Contour Analysis"):
            st.session_state.app_step = "analyze"
            st.experimental_rerun()
    elif canvas_result.json_data is not None:
        st.info("ℹ️ Click two points on the image (first rightmost, then leftmost).")

# --- STEP 3: CONTOUR ANALYSIS ---

if st.session_state.app_step == "analyze":
    rightmost_x = st.session_state.rightmost_point
    leftmost_x = st.session_state.leftmost_point

    if st.button("✅ Run Contour Analysis"):
        yellow_mask = st.session_state.yellow_mask
        orig_h, orig_w = st.session_state.orig_shape

        radius = 30
        angle_step = 10
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
        lx = leftmost_x[0]
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
            for angle in angles:
                rad = np.deg2rad(angle)
                x = int(round(cx + radius * np.cos(rad)))
                y = int(round(cy + radius * np.sin(rad)))
                dist = cv2.pointPolygonTest(main_contour, (x, y), True)
                if abs(dist) < 3:
                    return (x, y)
            return None

        for _ in range(max_steps):
            cx, cy = top_path[-1]
            next_pt = find_next_contour_point(cx, cy, radius, 270, 90)
            if not next_pt or next_pt[0] < lx:
                break
            top_path.append(next_pt)

        for _ in range(max_steps):
            cx, cy = bottom_path[-1]
            next_pt = find_next_contour_point(cx, cy, radius, 90, 270)
            if not next_pt or next_pt[0] < lx:
                break
            bottom_path.append(next_pt)

        st.success("✅ Contour analysis complete.")
        st.write(f"🔹 Top path points: {len(top_path)}")
        st.write(f"🔸 Bottom path points: {len(bottom_path)}")

        # === START: Midpoint Intersection Analysis ===
        # Convert yellow_mask to binary if not already
        _, binary_mask = cv2.threshold(yellow_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours again (for intersection analysis)
        contours_all, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare visualization image (color)
        visualization = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)

        # Use x-values from top_path and bottom_path for vertical line checks
        top_x_values = [pt[0] for pt in top_path]
        bottom_x_values = [pt[0] for pt in bottom_path]
        red_lines = top_x_values + bottom_x_values

        midpoints = []

        for x in red_lines:
            intersections = []
            for contour in contours_all:
                for i in range(len(contour) - 1):
                    pt1, pt2 = contour[i][0], contour[i + 1][0]
                    if pt1[0] <= x <= pt2[0] or pt2[0] <= x <= pt1[0]:
                        intersections.append((pt1[1], pt2[1]))

            # Sort intersections by y values
            if intersections:
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
                    group_2 = groups[1]
                    group_3 = groups[2]

                    y2_avg = np.mean([y for y, _ in group_2])
                    y3_avg = np.mean([y for y, _ in group_3])

                    midpoint_y = int((y2_avg + y3_avg) / 2)
                    midpoints.append((x, midpoint_y))

                    cv2.circle(visualization, (x, midpoint_y), 5, (0, 0, 255), -1)

                elif len(groups) == 3:
                    middle_group = groups[1]
                    middle_group_avg_y = np.mean([y for y, _ in middle_group])

                    midpoints.append((x, int(middle_group_avg_y)))

                    cv2.circle(visualization, (x, int(middle_group_avg_y)), 5, (0, 0, 255), -1)

                else:
                    last_midpoint_y = midpoints[-1][1] if midpoints else 0

                    first_group = groups[0]
                    first_group_avg_y = np.mean([y for y, _ in first_group])

                    last_group = groups[-1]
                    last_group_avg_y = np.mean([y for y, _ in last_group])

                    average_y = int((last_midpoint_y + first_group_avg_y + last_group_avg_y) / 3)

                    midpoints.append((x, average_y))

                    cv2.circle(visualization, (x, average_y), 5, (0, 0, 255), -1)

            color = (255, 0, 0) if not intersections else (0, 255, 0)
            cv2.line(visualization, (x, 0), (x, yellow_mask.shape[0]), color, 1)


        # Draw lines connecting midpoints
        if len(midpoints) > 1:
            for i in range(1, len(midpoints)):
                pt1 = midpoints[i - 1]
                pt2 = midpoints[i]
                cv2.line(visualization, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 255), 2)

        # Extend horizontal lines at ends
        if midpoints:
            leftmost_x_point, leftmost_y = midpoints[-1]
            cv2.line(visualization, (leftmost_x_point, leftmost_y), (0, leftmost_y), (255, 255, 0), 2)

            rightmost_x_point, rightmost_y = midpoints[0]
            cv2.line(visualization, (rightmost_x_point, rightmost_y), (rightmost_x_point + 500, rightmost_y), (255, 255, 0), 2)

        # Show visualization
        st.image(visualization, caption="Midpoint Intersection Analysis", use_column_width=True)

        # Additional confirmation checkmark below midpoint creation
        st.success("✅ Midpoint creation done.")

        # Store values needed for next step in session state
        st.session_state.yellow_mask_processed = yellow_mask
        st.session_state.top_x_values = top_x_values
        st.session_state.bottom_x_values = bottom_x_values
        st.session_state.midpoints = midpoints

    # Move the Next button outside the Run Contour Analysis button logic
    if st.button("➡️ Next: See Diameter Visualization and Graph"):
        st.session_state.app_step = "diameter"
        st.experimental_rerun()

# --- STEP 4: DIAMETER MEASUREMENT ---

if st.session_state.app_step == "diameter":
    
    st.title("Nerve Diameter Measurement")

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

    top_points = []
    bottom_points = []
    found_intersections = False

    # --- Find top points ---
    for i, x in enumerate(top_x_values):
        column = yellow_mask[:, x]
        nonzero_y = np.where(column > 0)[0]
        if len(nonzero_y) < 2:
            continue
        top_intersection = nonzero_y[0]
        found_intersections = True
        midpoint_y = midpoints[i][1] if i < len(midpoints) else height // 2
        if top_intersection < midpoint_y:
            top_points.append((x, top_intersection, i))
            cv2.circle(color_mask, (x, top_intersection), 6, (0, 255, 255), -1)  # yellow circle

    # --- Find bottom points ---
    for i, x in enumerate(bottom_x_values):
        column = yellow_mask[:, x]
        nonzero_y = np.where(column > 0)[0]
        if len(nonzero_y) < 2:
            continue
        bottom_intersection = nonzero_y[-1]
        found_intersections = True
        midpoint_y = midpoints[i][1] if i < len(midpoints) else height // 2
        if bottom_intersection > midpoint_y:
            bottom_points.append((x, bottom_intersection, i))
            cv2.circle(color_mask, (x, bottom_intersection), 6, (0, 255, 255), -1)  # yellow circle

    def estimate_tangent_slope(points, index, search_range=3):
        if len(points) < 2:
            return None
        x, y, _ = points[index]
        left_idx = max(0, index - search_range)
        right_idx = min(len(points) - 1, index + search_range)
        x1, y1, _ = points[left_idx]
        x2, y2, _ = points[right_idx]
        if x2 - x1 == 0:
            return None
        return (y2 - y1) / (x2 - x1)

    def find_mask_intersection(mask, start_x, start_y, angle, midpoint_y, is_top, max_length=300):
        dx = np.cos(angle)
        dy = np.sin(angle)
        for length in range(1, max_length):
            x = int(start_x + length * dx)
            y = int(start_y + length * dy)
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                return None
            if is_top and dy > 0 and y >= midpoint_y:
                return (x, y)
            if not is_top and dy < 0 and y <= midpoint_y:
                return (x, y)
            if mask[y, x] == 0:
                return (x, y)
        return None

    diameters_top = []
    diameters_bottom = []

    # Process top points
    for i, (x, y, section_idx) in enumerate(top_points):
        slope = estimate_tangent_slope(top_points, i)
        if slope is None:
            continue
        midpoint_y = midpoints[section_idx][1] if section_idx < len(midpoints) else height // 2
        perp_angle = np.arctan(-1/slope) if slope != 0 else np.pi/2
        intersection1 = find_mask_intersection(yellow_mask, x, y, perp_angle + np.pi,
                                              midpoint_y, True)
        intersection2 = find_mask_intersection(yellow_mask, x, y, perp_angle,
                                              midpoint_y, True)
        if intersection1 and intersection2:
            x1, y1 = intersection1
            x2, y2 = intersection2
            diameter = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            diameters_top.append((x, diameter))
            cv2.line(color_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Process bottom points
    for i, (x, y, section_idx) in enumerate(bottom_points):
        slope = estimate_tangent_slope(bottom_points, i)
        if slope is None:
            continue
        midpoint_y = midpoints[section_idx][1] if section_idx < len(midpoints) else height // 2
        perp_angle = np.arctan(-1/slope) if slope != 0 else np.pi/2
        intersection1 = find_mask_intersection(yellow_mask, x, y, perp_angle,
                                              midpoint_y, False)
        intersection2 = find_mask_intersection(yellow_mask, x, y, perp_angle + np.pi,
                                              midpoint_y, False)
        if intersection1 and intersection2:
            x1, y1 = intersection1
            x2, y2 = intersection2
            diameter = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            diameters_bottom.append((x, diameter))
            cv2.line(color_mask, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw midpoints
    for midpoint in midpoints:
        cv2.circle(color_mask, (int(midpoint[0]), int(midpoint[1])), 4, (255, 255, 0), -1)

    # Show the mask with diameter lines and midpoints in Streamlit
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    ax1.set_xlim(0, width)
    ax1.set_ylim(height, 0)
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_title('Nerve Diameter Measurements with Midpoint Constraint')
    st.pyplot(fig1)

    # Plot diameter vs x position
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if diameters_top:
        x_vals_top = [x for x, d in diameters_top]
        d_vals_top = [d for x, d in diameters_top]
        ax2.plot(x_vals_top, d_vals_top, 'go-', label='Top Nerve')
    if diameters_bottom:
        x_vals_bottom = [x for x, d in diameters_bottom]
        d_vals_bottom = [d for x, d in diameters_bottom]
        ax2.plot(x_vals_bottom, d_vals_bottom, 'ro-', label='Bottom Nerve')
    ax2.set_xlabel('Relative X Position on Nerve')
    ax2.set_ylabel('Diameter')
    ax2.set_title('Nerve Diameter vs X Position')
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)