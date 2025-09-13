import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import os

from depth_anything_v2.dpt import DepthAnythingV2


def capture_realsense_frame():
    """Capture one RGB + Depth frame from RealSense camera."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # e.g. 0.001 meters

    try:
        for _ in range(5):  # skip unstable first frames
            frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        rgb_image = np.asanyarray(color_frame.get_data())

        return rgb_image, depth_image, depth_scale
    finally:
        pipeline.stop()


def normalize_depth(depth):
    """Normalize depth map to 0-255 range for visualization."""
    depth = depth.astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
    return depth.astype(np.uint8)


def convert_da_to_meters(d_da, d_rs_meters):
    """Convert DA disparity to absolute depth in meters using RS depth."""
    mask = d_rs_meters > 0
    x = d_da[mask].flatten().astype(np.float32)
    y = (1.0 / d_rs_meters[mask].flatten().astype(np.float32))

    # Fit linear regression y ≈ a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    print(f"[Calibration] a={a:.6f}, b={b:.6f}")

    d_da_meters = 1.0 / (a * d_da.astype(np.float32) + b)
    d_da_meters = np.clip(d_da_meters, 0.1, 10.0)  # valid range [0.1m, 10m]
    return d_da_meters


def fuse_depths(d_rs_meters, d_da_meters):
    """Fuse RealSense and DepthAnything depth maps."""
    # Step 1: Edge detection on DA depth
    da_norm = cv2.normalize(d_da_meters, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(da_norm, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Step 2: Distance from edges → blending weight
    dist_from_edge = cv2.distanceTransform(cv2.bitwise_not(edges_dilated), cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist_from_edge, None, 0, 1.0, cv2.NORM_MINMAX)
    w = np.clip(dist_norm, 0, 1)

    # Step 3: Blend RS and DA
    d_fused = w * d_rs_meters + (1 - w) * d_da_meters
    return d_fused


# ========== Mouse click helper ==========
def click_event_factory(depth_map, name):
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                depth_val = float(depth_map[y, x])
                print(f"[{name}] Row={y}, Col={x}, Depth={depth_val:.3f} m")
    return click_event


def main():
    DEVICE = 'cpu'   # CPU only

    # Load Depth Anything (SMALL model)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    depth_anything = DepthAnythingV2(**model_configs['vits'])
    depth_anything.load_state_dict(
        torch.load("checkpoints/depth_anything_v2_vits.pth", map_location='cpu')
    )
    depth_anything = depth_anything.to(DEVICE).eval()

    # 1. Capture RGB + Depth from RealSense
    I_rgb, d_rs_raw, depth_scale = capture_realsense_frame()
    d_rs_meters = d_rs_raw.astype(np.float32) * depth_scale

    # 2. Run Depth Anything on RGB
    d_da = depth_anything.infer_image(I_rgb, 256)

    # 3. Convert DA disparity → meters
    d_da_meters = convert_da_to_meters(d_da, d_rs_meters)

    # 4. Fuse depths
    d_fused = fuse_depths(d_rs_meters, d_da_meters)

    # 5. Normalize maps for visualization
    d_rs_norm = normalize_depth(d_rs_meters)
    d_da_norm = normalize_depth(d_da)
    d_da_meters_norm = normalize_depth(d_da_meters)
    d_fused_norm = normalize_depth(d_fused)

    # 6. Save results
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/rgb.png", I_rgb)
    cv2.imwrite("output/d_rs_colormap.png",
                cv2.applyColorMap(d_rs_norm, cv2.COLORMAP_JET))
    cv2.imwrite("output/d_da_colormap.png",
                cv2.applyColorMap(d_da_norm, cv2.COLORMAP_JET))
    cv2.imwrite("output/d_da_meters_colormap.png",
                cv2.applyColorMap(d_da_meters_norm, cv2.COLORMAP_JET))
    cv2.imwrite("output/d_fused_colormap.png",
                cv2.applyColorMap(d_fused_norm, cv2.COLORMAP_JET))

    print("✅ Saved results in ./output folder")

    # 7. Show in separate windows
    cv2.namedWindow("RS Depth (colormap)")
    cv2.namedWindow("DA Depth (converted meters colormap)")
    cv2.namedWindow("Fused Depth (colormap)")

    cv2.setMouseCallback("RS Depth (colormap)", click_event_factory(d_rs_meters, "RealSense"))
    cv2.setMouseCallback("DA Depth (converted meters colormap)", click_event_factory(d_da_meters, "DepthAnything"))
    cv2.setMouseCallback("Fused Depth (colormap)", click_event_factory(d_fused, "Fused"))

    cv2.imshow("RGB Image", I_rgb)
    cv2.imshow("RS Depth (colormap)", cv2.applyColorMap(d_rs_norm, cv2.COLORMAP_JET))
    cv2.imshow("DA Depth (relative colormap)", cv2.applyColorMap(d_da_norm, cv2.COLORMAP_JET))
    cv2.imshow("DA Depth (converted meters colormap)", cv2.applyColorMap(d_da_meters_norm, cv2.COLORMAP_JET))
    cv2.imshow("Fused Depth (colormap)", cv2.applyColorMap(d_fused_norm, cv2.COLORMAP_JET))

    print("👉 Click on any window to inspect depth values in meters (ESC to quit).")

    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
