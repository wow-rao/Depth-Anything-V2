import numpy as np
import pyrealsense2 as rs
import torch
import os
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm

from depth_anything_v2.dpt import DepthAnythingV2


def capture_realsense_frame():
    """Capture one RGB + Depth frame from RealSense camera."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # Use RGB format directly to avoid BGR to RGB conversion
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # e.g. 0.001 meters

    try:
        # Allow auto-exposure to settle
        for _ in range(5):
            frames = pipeline.wait_for_frames()
        
        align = rs.align(rs.stream.color)
        frameset = align.process(frames)

        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        rgb_image = np.asanyarray(color_frame.get_data())

        return rgb_image, depth_image, depth_scale
    finally:
        pipeline.stop()


def normalize_depth(depth):
    """Normalize depth map to 0-255 range for visualization."""
    depth = depth.astype(np.float32)
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max > depth_min:
        # Normalize to 0-1, then scale to 0-255
        normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        return (normalized * 255.0).astype(np.uint8)
    return np.zeros_like(depth, dtype=np.uint8)


def convert_da_to_meters(d_da, d_rs_meters):
    """Convert DA disparity to absolute depth in meters using RS depth."""
    # Mask out invalid RealSense depth values (which are 0)
    mask = d_rs_meters > 0
    
    # Get corresponding values from DA disparity and RS depth
    x = d_da[mask].flatten().astype(np.float32)
    # Convert RealSense depth to inverse depth (disparity)
    y = (1.0 / d_rs_meters[mask].flatten().astype(np.float32))

    # Fit a linear regression model: y ≈ a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    print(f"[Calibration] a={a:.6f}, b={b:.6f}")

    # Apply the learned transformation to the entire DA disparity map
    # Using np.maximum to avoid division by zero or negative depths
    d_da_meters = 1.0 / np.maximum(a * d_da.astype(np.float32) + b, 1e-6)
    
    # Clip to a reasonable depth range
    d_da_meters = np.clip(d_da_meters, 0.1, 10.0)
    return d_da_meters


def create_gaussian_kernel(kernel_size, sigma):
    """Creates a 2D Gaussian kernel using PyTorch."""
    # Create a 1D tensor of coordinates
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0

    # Compute the 1D Gaussian distribution
    g = torch.exp(-(coords**2) / (2 * sigma**2))

    # Create the 2D kernel by taking the outer product
    kernel_2d = torch.outer(g, g)

    # Normalize the kernel so that it sums to 1
    kernel_2d /= torch.sum(kernel_2d)

    # Reshape for conv2d: [out_channels, in_channels, H, W]
    return kernel_2d.reshape(1, 1, kernel_size, kernel_size)


def fuse_depths_torch(d_rs_meters, d_da_meters):
    """
    Fuse RealSense and DepthAnything depth maps using PyTorch.
    This function is free of OpenCV and Torchvision dependencies.
    """
    # Convert numpy arrays to torch tensors, adding batch and channel dimensions
    d_rs_tensor = torch.from_numpy(d_rs_meters).unsqueeze(0).unsqueeze(0).float()
    d_da_tensor = torch.from_numpy(d_da_meters).unsqueeze(0).unsqueeze(0).float()
    
    device = d_rs_tensor.device

    # --- Step 1: Edge detection on DA depth (Sobel filter) ---
    da_norm = (d_da_tensor - d_da_tensor.min()) / (d_da_tensor.max() - d_da_tensor.min() + 1e-6)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)

    grad_x = F.conv2d(da_norm, sobel_x, padding=1)
    grad_y = F.conv2d(da_norm, sobel_y, padding=1)
    
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
    edges = (grad_mag > 0.3).float()

    edges_dilated = F.max_pool2d(edges, kernel_size=3, stride=1, padding=1)

    # --- Step 2: Create blending weight (custom Gaussian blur) ---
    inverted_edges = 1 - edges_dilated
    
    # Create and apply Gaussian kernel for blurring
    kernel_size, sigma = 15, 5
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(device)
    padding = kernel_size // 2
    w = F.conv2d(inverted_edges, gaussian_kernel, padding=padding)
    
    # --- Step 3: Blend RS and DA ---
    d_fused_tensor = w * d_rs_tensor + (1 - w) * d_da_tensor

    return d_fused_tensor.squeeze().cpu().numpy()


def save_colormap(path, depth_map_normalized):
    """Saves a normalized (0-255) depth map as a color image."""
    if depth_map_normalized.dtype != np.uint8:
        raise ValueError("Input depth map must be of type uint8.")
        
    colored_image_rgba = cm.jet(depth_map_normalized / 255.0)
    colored_image_rgb = (colored_image_rgba[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored_image_rgb).save(path)


def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Load Depth Anything (SMALL model)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    depth_anything = DepthAnythingV2(**model_configs['vits'])
    checkpoint_path = "checkpoints/depth_anything_v2_vits.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        print("Please download the vits model from the Depth Anything V2 repository.")
        return
        
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    depth_anything = depth_anything.to(DEVICE).eval()

    # 1. Capture RGB + Depth from RealSense
    print("Capturing frame from RealSense camera...")
    I_rgb, d_rs_raw, depth_scale = capture_realsense_frame()
    d_rs_meters = d_rs_raw.astype(np.float32) * depth_scale
    print("Frame captured.")

    # 2. Run Depth Anything on RGB
    print("Running Depth Anything inference...")
    d_da = depth_anything.infer_image(I_rgb, 256)
    print("Inference complete.")

    # 3. Convert DA disparity to meters
    print("Calibrating Depth Anything output to meters...")
    d_da_meters = convert_da_to_meters(d_da, d_rs_meters)
    print("Calibration complete.")

    # 4. Fuse depths
    print("Fusing depth maps...")
    d_fused = fuse_depths_torch(d_rs_meters, d_da_meters)
    print("Fusion complete.")

    # 5. Normalize all depth maps for visualization
    d_rs_norm = normalize_depth(d_rs_meters)
    d_da_norm = normalize_depth(d_da)
    d_da_meters_norm = normalize_depth(d_da_meters)
    d_fused_norm = normalize_depth(d_fused)

    # 6. Save results to the output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving results to '{output_dir}' directory...")
    
    Image.fromarray(I_rgb).save(os.path.join(output_dir, "rgb.png"))
    
    save_colormap(os.path.join(output_dir, "d_rs_colormap.png"), d_rs_norm)
    save_colormap(os.path.join(output_dir, "d_da_colormap.png"), d_da_norm)
    save_colormap(os.path.join(output_dir, "d_da_meters_colormap.png"), d_da_meters_norm)
    save_colormap(os.path.join(output_dir, "d_fused_colormap.png"), d_fused_norm)

    print(f"All results saved successfully in the '{output_dir}' folder.")


if __name__ == "__main__":
    main()
