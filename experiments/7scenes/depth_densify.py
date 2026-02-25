import collections
import pdb

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_3_tensor = torch.tensor(FULL_KERNEL_3, dtype=torch.float32).cuda()
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_5_tensor = torch.tensor(FULL_KERNEL_5, dtype=torch.float32).cuda()

FULL_KERNEL_7 = np.ones((7, 7), np.uint8)

FULL_KERNEL_7_tensor = torch.tensor(FULL_KERNEL_7, dtype=torch.float32).cuda()

FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)
FULL_KERNEL_31_tensor = torch.tensor(FULL_KERNEL_31, dtype=torch.float32).cuda()

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)
DIAMOND_KERNEL_7_tensor = torch.tensor(DIAMOND_KERNEL_7, dtype=torch.float32).cuda()

# 9x9 diamond kernel
DIAMOND_KERNEL_9 = np.asarray(
    [
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
    ], dtype=np.uint8)
DIAMOND_KERNEL_9_tensor = torch.tensor(DIAMOND_KERNEL_9, dtype=torch.float32).cuda()

def get_padded_map(map, custom_kernel):
    h = map.shape[0]
    w = map.shape[1]
    N = custom_kernel.shape[0]
    n = int((N - 1) / 2)
    padded_depth = torch.nn.functional.pad(map, (n, n, n, n), mode='constant', value=0.)
    depth_list = []
    for i in range(N):
        for j in range(N):
            if custom_kernel[i][j] == 1:
                depth_list.append(padded_depth[i:i + h, j:j + w].unsqueeze(-1))
    depth_list = torch.cat(depth_list, -1)
    return depth_list

def dilate_tensor(depth_map, custom_kernel):
    depth_list = get_padded_map(depth_map, custom_kernel)
    depth_map = torch.max(depth_list, -1)[0]

    return depth_map

def close_tensor(depth_map, custom_kernel_erode, custom_kernel_dilate):
    depth_list = get_padded_map(depth_map, custom_kernel_erode)
    depth_map = torch.min(depth_list, -1)[0]
    depth_list = get_padded_map(depth_map, custom_kernel_dilate)
    depth_map = torch.max(depth_list, -1)[0]
    return depth_map

def median_tensor(depth_map, kernel_size):
    custom_kernel = torch.ones((kernel_size, kernel_size)).cuda()
    depth_list = get_padded_map(depth_map, custom_kernel)
    depth_map = torch.median(depth_list, -1)[0]

    return depth_map

def save_depth(dpt, save_path):
    scale = 255 / dpt.max()
    tmp = dpt * scale
    if tmp.device.type == 'cuda':
        tmp = tmp.detach().cpu().numpy()
    # tmp = dpt
    tmp = tmp.astype('uint8')
    tmp = Image.fromarray(tmp)

    tmp.save(save_path)
    return tmp




def fast_differentiable_fill(depth_map, max_depth=15.0):
    """Fast, in-place depth completion.

        Args:
            depth_map: projected depths
            max_depth: max depth value for inversion
            custom_kernel: kernel to apply initial dilation
            extrapolate: whether to extrapolate by extending depths to top of
                the frame, and applying a 31x31 full kernel dilation
            blur_type:
                'bilateral' - preserves local structure (recommended)
                'gaussian' - provides lower RMSE

        Returns:
            depth_map: dense depth map
        """
    img_dir = "imgs"
    # Invert
    valid_pixels = (depth_map > 0.1)

    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    # Dilate

    # save_depth(depth_map, f"{img_dir}/point.png")

    depth_map = dilate_tensor(depth_map, DIAMOND_KERNEL_7_tensor)

    # save_depth(depth_map, f"{img_dir}/dilate.png")

    # Hole closing
    depth_map = close_tensor(depth_map, FULL_KERNEL_3_tensor, FULL_KERNEL_5_tensor)
    # save_depth(depth_map, f"{img_dir}/close.png")

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)

    dilated = dilate_tensor(depth_map, FULL_KERNEL_7_tensor)
    depth_map[empty_pixels] = dilated[empty_pixels]
    # save_depth(depth_map, f"{img_dir}/fill_empty.png")

    # Median blur
    depth_map = median_tensor(depth_map, 5)
    # save_depth(depth_map, f"{img_dir}/median.png")

    # Gaussian blur
    valid_pixels = (depth_map > 0.1).detach()


    transform = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
    # 应用高斯模糊变换
    blurred = transform(depth_map.unsqueeze(0)).squeeze()

    # depth_map[valid_pixels] = blurred[valid_pixels]
    depth_map = torch.where(valid_pixels, blurred, depth_map)


    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    # save_depth(depth_map, f"{img_dir}/gaussian.png")

    return depth_map

def fill_in_fast(depth_map, max_depth=15.0, custom_kernel=DIAMOND_KERNEL_7,
                 extrapolate=False, blur_type='gaussian'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """
    if depth_map.device.type == 'cuda':
        depth_map = depth_map.detach.cpu().numpy()
    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    # Dilate
    save_depth(depth_map, "gt/point.png")

    depth_map = cv2.dilate(depth_map, custom_kernel)
    save_depth(depth_map, "gt/dilate.png")

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)
    save_depth(depth_map, "gt/close.png")

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]
    save_depth(depth_map, "gt/fill_empty.png")

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)
    save_depth(depth_map, "gt/median.png")

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]


    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    save_depth(depth_map, "gt/gaussian.png")
    return depth_map


def fill_in_multiscale(depth_map, 
                       min_depth=0.1,
                       second_depth=15.0,
                       third_depth=30.0,
                       max_depth=100.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > min_depth) & (depths_in <= second_depth)
    valid_pixels_med = (depths_in > second_depth) & (depths_in <= third_depth)
    valid_pixels_far = (depths_in > third_depth)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.1)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.1)
    valid_pixels_med = (dilated_med > 0.1)
    valid_pixels_far = (dilated_far > 0.1)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.1)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool_)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.1)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool_)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out
    
    depths_out[depths_out<0.1] = 255
    
    return depths_out


name2densefunc = {
    'fast': fill_in_fast,
    'multiscale': fill_in_multiscale,
    "fast_differentiable": fast_differentiable_fill,
}

if __name__ == '__main__':
    sparse = Image.open('depth.png')
    sparse = np.array(sparse)
    dense = fill_in_fast(sparse)
    depth_image_normalized = dense.astype('uint8')  # Normalize to [0, 255]
    img = Image.fromarray(depth_image_normalized)
    img.save('dense.png')