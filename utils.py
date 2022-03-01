import argparse
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A
import tensorflow as tf
import random


cells_dict = {'[1.0, 0.0, 0.0, 0.0]': [[0, 0, 0, 0]],
              '[0.0, 1.0, 0.0, 0.0]': [[1, 1, 1, 1]],
              '[0.0, 0.0, 1.0, 0.0]': [[2, 2, 2, 2]],
              '[0.0, 0.0, 0.0, 1.0]': [[3, 3, 3, 3]],
              '[1.0, 1.0, 0.0, 0.0]': [[0, 0, 1, 1]],
              '[1.0, 0.0, 1.0, 0.0]': [[0, 0, 2, 2]],
              '[1.0, 0.0, 0.0, 1.0]': [[0, 0, 3, 3]],
              '[0.0, 1.0, 1.0, 0.0]': [[1, 1, 2, 2]],
              '[0.0, 1.0, 0.0, 1.0]': [[1, 1, 3, 3]],
              '[0.0, 0.0, 1.0, 1.0]': [[2, 2, 3, 3]],
              '[1.0, 1.0, 1.0, 0.0]': [[0, 0, 1, 2]],
              '[1.0, 1.0, 0.0, 1.0]': [[0, 1, 1, 3]],
              '[1.0, 0.0, 1.0, 1.0]': [[0, 2, 2, 3]],
              '[0.0, 1.0, 1.0, 1.0]': [[1, 2, 3, 3]],
              '[1.0, 1.0, 1.0, 1.0]': [[0, 1, 2, 3]]}

organ_dict = {'[1.0, 0.0, 0.0, 0.0]': [[0]],
              '[0.0, 1.0, 0.0, 0.0]': [[1]],
              '[0.0, 0.0, 1.0, 0.0]': [[2]],
              '[0.0, 0.0, 0.0, 1.0]': [[3]]}

# --------------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, help='final resolution')
    parser.add_argument('--wgan_penalty_const', type=int, help='wasserstein penalty constant')
    parser.add_argument('--discriminator_iterations', type=int, help='number of discriminator training iterations')
    parser.add_argument('--iterations', type=int, help='number of initial iterations')
    parser.add_argument('--interval', type=int, help='how many iterations to show results')
    parser.add_argument('--max_log2res', type=int, help='load resolution')
    parser.add_argument('--grow_model', type=int, help='grow loaded model')
    parser.add_argument('--wandb_key', type=str, help='wandb key')
    parser.add_argument("--data_path", "-d", type=str, default="../../dataset/",
                        help="Path to directory, where data is stored.")

    args = parser.parse_args()

    return args


# --------------------------------------------------------------------------------
def plot_images(images, log2_res, fname=''):
    organs = {0: 'pľúca', 1: 'obličky', 2: 'prsia', 3: 'prostata'}
    scales = {2: 0.5,
              3: 1,
              4: 2,
              5: 3,
              6: 4,
              7: 5,
              8: 6,
              9: 7,
              10: 8}
    scale = scales[log2_res]

    grid_col = 4
    grid_row = 4

    f, axarr = plt.subplots(grid_row, grid_col, figsize=(grid_col * scale, grid_row * scale))

    for row in range(grid_row):
        ax = axarr if grid_row == 1 else axarr[row]
        for col in range(grid_col):
            ax[col].imshow(images[row * grid_col + col])
            ax[col].axis('off')

    flag = 'maska' if 'masks' in fname else 'obrazok'
    wandb.log({f"{2 ** log2_res} {flag}": wandb.Image(f)})
    # plt.show()
    plt.close(f)


# --------------------------------------------------------------------------------
def detect_cell_types(mask):
    labels = [0., 0., 0., 0.]

    c = set()
    d = mask.getdata()

    for item in d:
        c.add(item)
        if (255, 0, 0) in c:
            labels[0] = 1.
        if (255, 255, 0) in c:
            labels[1] = 1.
        if (0, 255, 0) in c:
            labels[2] = 1.
        if (0, 0, 255) in c:
            labels[3] = 1.

    return labels


# --------------------------------------------------------------------------------

def load_real_samples(df, batch_size, data_path, log2_res, n_rows):
    samples = []
    cells = []
    organ = []
    indices = random.sample(range(0, n_rows), batch_size)
    res = 2**log2_res
    transform = A.Compose([
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.Rotate(limit=10, p=1)
    ])

    for idx in indices:
        image = Image.open(f'{data_path}/v4/{res}/images/' + df['0'].loc[idx])
        mask = Image.open(f'{data_path}/v4/{res}/masks/' + df['0'].loc[idx])

        image_np = np.array(image)
        image_np = (image_np - 127.5) / 127.5

        mask_np = np.array(mask)
        mask_np = (mask_np - 127.5) / 127.5

        transformed = transform(image=image_np[:, :, 0:3], mask=mask_np)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        samples.append(np.concatenate((transformed_image, transformed_mask), axis=2))
        cells.append(cells_dict[df['1'].loc[idx]])
        organ.append(organ_dict[df['2'].loc[idx]])

    return np.array(samples).astype(np.float32), np.array(cells), np.array(organ)

# --------------------------------------------------------------------------------
def generate_labels(batch_size):
    labels = np.zeros((batch_size, 4))

    for i in range(batch_size):
        res = [random.randrange(0, 4, 1) for i in range(random.randint(1, 4))]

        for r in res:
            labels[i][r] = 1.0

    return labels


def generate_organ(batch_size, organ):
    organs = np.zeros((batch_size, 4))

    for i in range(batch_size):
        organs[i][organ] = 1.0

    return organs


def generate_organs(batch_size):
    organs = np.zeros((batch_size, 4))

    for i in range(batch_size):
        r = random.randint(0, 3)
        organs[i][r] = 1.0

    return organs


def generate_noise(batch_size, log2_resolution):
    return [tf.random.normal((batch_size, 2 ** res, 2 ** res, 1)) for res in range(2, log2_resolution + 1)]
# --------------------------------------------------------------------------------


def gradient_loss(grad, penalty_const):
    loss = tf.square(grad)
    loss = tf.reduce_sum(loss, axis=np.arange(1, len(loss.shape)))
    loss = tf.sqrt(loss)
    loss = tf.reduce_mean(tf.square(loss - 1))
    loss = penalty_const * loss
    return loss
