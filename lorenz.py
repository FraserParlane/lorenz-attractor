"""
An animation of the Lorenz attractor
"""
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import imageio
import shutil
import os


def lorenz_iterator(
        point: np.array,
        s: float = 10,
        r: float = 28,
        b: float = 2.667,
) -> np.array:
    """
    Iterate on the Lorenz equation
    :param point: The previous point
    :param s: Lorenz parameter
    :param r: Lorenz parameter
    :param b: Lorenz parameter
    :return: Vector to next point
    """

    x, y, z = point
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


def lorenz_generator(
        init: np.array = np.array([0, 1, 1.05]),
        step: float = 0.01,
        line_res: int = 500,
) -> np.array:
    """
    Generate a Lorenz path
    :param init: Initial point
    :param step: Step size
    :param line_res: Number of steps
    :return: x, y, z array of points
    """

    # Initialize array
    points = np.empty((line_res, 3))
    points[0] = init

    # Iterate
    for i in range(line_res - 1):
        dot = lorenz_iterator(point=points[i])
        points[i+1] = points[i] + dot * step

    # Return values
    return points


def animate(
        n_traces: int = 5,
        filename: str = 'animation',
        init_point: np.array = np.array([0, 1, 1.05]),
        fps: int = 25,
        dur_sec: int = 20,
        tail_length: int = 200,
        line_res: int = 5000,
        step: float = 0.005,
        colorscale: str = 'viridis',
        bg_color: str = '#22272E',
) -> None:

    # First, prepare the frames folder
    frames_path = os.path.join(os.getcwd(), 'frames')
    if os.path.exists(frames_path):
        shutil.rmtree(frames_path)
    os.mkdir(frames_path)

    # Generate the initial points such that they are randomly perturbed.
    init_points = np.repeat(init_point[None, :], n_traces, axis=0)
    init_points = np.random.normal(init_points, scale=0.1)

    # Calculate the Lorentz points
    lor_points = []
    for init_point in init_points:
        lor_point = lorenz_generator(
            init=init_point,
            line_res=line_res,
            step=step,
        )
        lor_points.append(lor_point)
    lor_points = np.array(lor_points)

    # Calculate the bounds of the data for formatting the plot
    min_x = lor_points[:, :, 0].min()
    max_x = lor_points[:, :, 0].max()
    min_y = lor_points[:, :, 2].min()
    max_y = lor_points[:, :, 2].max()
    span_x = max_x - min_x
    span_y = max_y - min_y
    buffer = 0.01
    plot_min_x = min_x - span_x * buffer
    plot_max_x = max_x + span_x * buffer
    plot_min_y = min_y - span_y * buffer
    plot_max_y = max_y + span_y * buffer

    # Given the FPS and duration, determine how many points to plot
    n_frames = fps * dur_sec
    res_frame = int(line_res / n_frames)

    # Store the frame paths
    frame_paths = []

    # Iterate through each frame and save to a png
    for i in tqdm(range(2, n_frames)):

        # Determine the start and stop indexes of the frame
        stop = i * res_frame
        start = max(0, stop - tail_length)

        # Create the plotting objects
        figure: plt.Figure = plt.figure(
            facecolor=bg_color,
        )
        ax: plt.Axes = figure.add_subplot()

        # Iterate through each of the traces
        for j in range(n_traces):

            # Select the x and y data
            x = lor_points[j, start:stop, 0]
            y = lor_points[j, start:stop, 2]

            # Generate the gradient line collection
            cols = np.linspace(0, 1, len(x))
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=colorscale)
            lc.set_array(cols)
            ax.add_collection(lc)

        # Format
        ax.set_xlim(plot_min_x, plot_max_x)
        ax.set_ylim(plot_min_y, plot_max_y)
        ax.axis('off')
        ax.set_facecolor(bg_color)
        figure.subplots_adjust(
            left=0.05,
            right=0.95,
            bottom=0.05,
            top=0.95,
        )

        # Save to file
        frame_path = f'frames/{i:04}.png'
        figure.savefig(frame_path)
        frame_paths.append(f'frames/{i:04}.png')
        plt.close(figure)

    # Convert the frames into an animated gif
    frames = []
    for frame_path in tqdm(frame_paths):
        frames.append(imageio.v2.imread(frame_path))
    imageio.mimsave(
        f'{filename}.gif',
        frames,
        duration=1 / fps,
    )


if __name__ == '__main__':
    slow_kwargs = {
        'n_traces': 10,
        'filename': 'animation',
        'init_point': np.array([0, 1, 1.05]),
        'fps': 25,
        'dur_sec': 20,
        'tail_length': 200,
        'line_res': 5000,
        'step': 0.004,
        'colorscale': 'viridis',
    }

    fast_kwargs = {
        'n_traces': 5,
        'filename': 'animation',
        'init_point': np.array([0, 1, 1.05]),
        'fps': 10,
        'dur_sec': 10,
        'tail_length': 200,
        'line_res': 1000,
        'step': 0.01,
        'colorscale': 'magma',
        'bg_color': '#000000',
    }

    animate(**slow_kwargs)
