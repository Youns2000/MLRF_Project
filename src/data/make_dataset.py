# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import pickle
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def unpickle(file):
    """ Load pickled file """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def flatten_images(images):
    """ Flatten 32*32*3 images into 3072-dimensional vectors """
    return images.reshape(images.shape[0], -1)


def load_and_transform_data(input_filepath, filename):
    """ Load and transform a single data batch """
    file = Path(input_filepath) / filename
    if not file.is_file():
        logger = logging.getLogger(__name__)
        logger.error(
            f"File {filename} not found in directory {input_filepath}")
        return None, None
    data_batch = unpickle(file)
    images = flatten_images(data_batch[b'data'])
    labels = np.array(data_batch[b'labels'])
    return images, labels


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    all_images = []
    all_labels = []
    for i in range(1, 6):
        filename = f'data_batch_{i}'
        images, labels = load_and_transform_data(input_filepath, filename)

        all_images.append(images)
        all_labels.append(labels)

    test_images, test_labels = load_and_transform_data(
        input_filepath, 'test_batch')

    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    os.makedirs(output_filepath, exist_ok=True)
    with open(Path(output_filepath) / 'images.npy', 'wb') as f:
        np.save(f, all_images)
    with open(Path(output_filepath) / 'labels.npy', 'wb') as f:
        np.save(f, all_labels)
    with open(Path(output_filepath) / 'test_images.npy', 'wb') as f:
        np.save(f, test_images)
    with open(Path(output_filepath) / 'test_labels.npy', 'wb') as f:
        np.save(f, test_labels)

    logger.info('Data processed and saved')

    meta_batch = unpickle(Path(input_filepath) / 'batches.meta')
    with open(Path(output_filepath) / 'label_names.pkl', 'wb') as f:
        pickle.dump(meta_batch, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()
