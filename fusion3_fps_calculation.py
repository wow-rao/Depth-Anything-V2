import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import os
import time

from depth_anything_v2.dpt import DepthAnythingV2


def init_realsense():
    """Initialize RealSense pipeline (keep running for speed)."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # skip unstable frames at startup
    for _ in range(5):
        pipeline.wait_for_frames()
    return pipeline


def get_realsense_frame(pipeline):
    """Capture one RGB + Depth frame from RealSense camera (pipeline must be running)."""
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    rgb_image = np.asanyarray(color_frame.get_data())
    return rgb_image, depth_image


def normalize_depth(depth):
    """Normalize depth map to 0-255 range."""
    depth = depth.astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
    return depth.astype(np.uint8)


def confidence_fusion(depth_rs, depth_da, conf_rs, conf_da):
    """Fuse depth maps using confidence-based weighting."""
    depth_rs = depth_rs.astype(np.float32)
    depth_da = depth_da.astype(np.float32)
    conf_rs = conf_rs.astype(np.float32)
    conf_da = conf_da.astype(np.float32)

    fused = (depth_rs * conf_rs + depth_da * conf_da) / (conf_rs + conf_da + 1e-6)
    return fused.astype(np.uint8)


def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on:", DEVICE)

    # Load Depth Anything (SMALL model)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    depth_anything = DepthAnythingV2(**model_configs['vits'])
    depth_anything.load_state_dict(
        torch.load("checkpoints/depth_anything_v2_vits.pth", map_location='cpu')
    )
    depth_anything = depth_anything.to(DEVICE).eval()

    # Init RealSense once
    pipeline = init_realsense()

    os.makedirs("fusion_output", exist_ok=True)

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            loop_start = time.time()

            # 1. Capture
            rgb_image, depth_rs = get_realsense_frame(pipeline)

            # 2. DepthAnything inference
            depth_da = depth_anything.infer_image(rgb_image, 256)

            # 3. Normalize
            depth_rs_norm = normalize_depth(depth_rs)
            depth_da_norm = normalize_depth(depth_da)

            # 4. Confidence maps
            conf_rs = (depth_rs_norm > 0).astype(np.float32) * 0.9
            conf_da = cv2.Laplacian(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
            conf_da = cv2.convertScaleAbs(conf_da)
            conf_da = cv2.normalize(conf_da.astype(np.float32), None, 0.2, 0.8, cv2.NORM_MINMAX)

            # 5. Fusion
            fused_depth = confidence_fusion(depth_rs_norm, depth_da_norm, conf_rs, conf_da)

            # ---- OPTIONAL: Save every 30th frame (to avoid slowing down) ----
            if frame_count % 30 == 0:
                cv2.imwrite("fusion_output/rgb.png", rgb_image)
                cv2.imwrite("fusion_output/depth_fused.png", fused_depth)
                cv2.imwrite("fusion_output/depth_fused_colormap.png",
                            cv2.applyColorMap(fused_depth, cv2.COLORMAP_JET))

            frame_count += 1
            loop_time = (time.time() - loop_start) * 1000  # ms
            fps = 1000.0 / loop_time if loop_time > 0 else 0
            print(f"Frame {frame_count}: {loop_time:.2f} ms ({fps:.1f} FPS)")

            # ---- For testing: stop after 200 frames ----
            if frame_count >= 200:
                break

    finally:
        pipeline.stop()

    total_time = time.time() - t0
    avg_fps = frame_count / total_time
    print(f"\n✅ Average FPS over {frame_count} frames: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
