import numpy as np
import torch

from typing import Union
from .logger import logger


def calculate_hz(sensor_name: str, timestamps: Union[list, np.array, torch.Tensor]) -> None:
    """Calculate Hz of Sensor Data."""
    length = timestamps[-1] - timestamps[0]
    average_timestep = length / len(timestamps)
    hz = 1 / average_timestep
    logger.info(f'{sensor_name} data: {length} Sec, {hz} Hz')
    return hz


def find_nearest_index(array: np.ndarray, time):  # array of timesteps, time to search for
    """Find closest time in array, that has already passed."""
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    elif isinstance(array, list):
        array = np.array(array)
    diff_arr = array - time
    idx = np.where(diff_arr <= 0, diff_arr, -np.inf).argmax()
    # [-0.02 +0.02 +2] becomes  [-0.02  -inf -inf]
    return idx


def sample_from_frequency(timestamps: Union[np.array, torch.Tensor], dt=float) -> Union[np.array]:
    """Sample the timestamps at a given frequency.

    Args:
        timestamps (Union[list, np.array]): The timestamps to sample.
        dt (float): The frequency to sample at x hz
    Returns:
        Union[list, np.array]: The sampled timestamps.
    """
    ori_type = type(timestamps)
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.cpu().numpy()
    elif isinstance(timestamps, list):
        timestamps = np.array(timestamps)

    logger.info(f'Original data: {len(timestamps)} points')
    calculate_hz('Origin data', timestamps)

    hz = 1 / dt
    logger.info(f'Sampling at {dt} Sec, {hz} Hz')
    t = np.arange(timestamps[0], timestamps[-1], dt)
    sampled_indices = []
    for k in range(len(t)):
        sampled_counter = find_nearest_index(timestamps, t[k])
        sampled_indices.append(sampled_counter)

    sampled_indices = np.array(sampled_indices).astype(int)
    sampled_timestamps = timestamps[sampled_indices]

    logger.info(f'Sampled data: {len(sampled_timestamps)} points')
    calculate_hz('Sampled', sampled_timestamps)
    if ori_type == torch.Tensor:
        sampled_timestamps = torch.from_numpy(sampled_timestamps)
    return sampled_indices, sampled_timestamps
