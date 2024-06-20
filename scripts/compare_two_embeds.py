import torch
import logging
from torch.nn import functional as Fun
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig()
logger = logging.getLogger()


def get_similarity(view1, view2):
    norm1 = torch.sum(torch.square(view1), dim=1)
    norm1 = norm1.reshape(-1, 1)
    norm2 = torch.sum(torch.square(view2), dim=1)
    norm2 = norm2.reshape(1, -1)
    similarity = norm1 + norm2 - 2.0 * \
        torch.matmul(view1, view2.transpose(1, 0))
    similarity = -1.0 * torch.max(similarity, torch.zeros(1).cuda())

    return similarity


def decision_offset(view1, view2, label):
    logger.debug(f'view1.shape: {view1.shape}')
    logger.debug(f'view2.shape: {view2.shape}')

    sim_12 = get_similarity(view1, view2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_12.cpu().numpy(),
                annot=True, cmap="viridis", cbar=True)

    # Save the heatmap to a PNG file
    plt.title("sim_12")
    plt.savefig("sim_12.png")
    plt.show()

    softmaxed_sim_12 = Fun.softmax(sim_12, dim=1)

    logger.debug(f'softmaxed_sim_12.shape: {softmaxed_sim_12.shape}')
    logger.debug(f'softmaxed_sim_12: {softmaxed_sim_12}')

    plt.figure(figsize=(10, 8))
    sns.heatmap(softmaxed_sim_12.cpu().numpy(),
                annot=True, cmap="viridis", cbar=True)

    # Save the heatmap to a PNG file
    plt.title("softmaxed_sim_12")
    plt.savefig("softmaxed_sim_12.png")
    plt.show()

    ground = (torch.tensor(
        [i * 1.0 for i in range(view1.size(0))]).cuda()).reshape(-1, 1)

    predict = softmaxed_sim_12.argmax(dim=1)

    logger.debug(f'predict: {predict}')

    length1 = ground.size(0)

    frames = []

    for i in range(length1):
        p = predict[i].item()
        g = ground[i][0].item()

        frame_error = (p - g)
        frames.append(frame_error)

    logger.debug(f'len(frames): {len(frames)}')
    logger.debug(f'frames: {frames}')

    median_frames = np.median(frames)
    mean_frames = np.average(frames)

    num_frames_median = math.floor(median_frames)
    num_frames_mean = math.floor(mean_frames)

    result_median = abs(num_frames_median - label)
    result_mean = abs(num_frames_mean - label)

    return {
        'result_median': result_median,
        'result_mean': result_mean
    }


def load_and_sort_tensors(file_path):
    # Load the tensor file
    tensor_dict = torch.load(file_path)

    # Sort the keys based on the frame number extracted from the key
    filtered_keys = filter(lambda x: x.split(
        '.')[0].isdigit(), tensor_dict.keys())
    sorted_keys = sorted(filtered_keys,
                         key=lambda x: int(x.split('.')[0]))

    # Stack the tensors in the sorted order
    sorted_tensors = torch.stack([tensor_dict[key] for key in sorted_keys])

    return sorted_tensors


# Paths to your tensor files
file_path1 = '/home/yosubs/koa_scratch/S001C001P001R001A001.pth'
file_path2 = '/home/yosubs/koa_scratch/S001C002P001R001A001.pth'

# Load and sort tensors from each file
view1 = load_and_sort_tensors(file_path1)
view2 = load_and_sort_tensors(file_path2)

# Print the shapes of the resulting tensors
print("Shape of sorted tensor from file 1:", view1.shape)
print("Shape of sorted tensor from file 2:", view2.shape)

print(decision_offset(view1, view2, 30))
