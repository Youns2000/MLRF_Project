# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import pickle
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.decomposition import PCA


def unpickle(file):
    """ Load pickled file """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_and_transform_data(input_filepath, filename):
    """ Load and transform a single data batch """
    file = Path(input_filepath) / filename
    if not file.is_file():
        logger = logging.getLogger(__name__)
        logger.error(
            f"File {filename} not found in directory {input_filepath}")
        return None, None
    data_batch = unpickle(file)
    images = data_batch[b'data']
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
    pca = PCA(n_components=100)
    pca.fit(all_images)
    all_images = pca.transform(all_images)
    test_images = pca.transform(test_images)

    os.makedirs(output_filepath, exist_ok=True)
    with open(Path(output_filepath) / 'images_pca.npy', 'wb') as f:
        np.save(f, all_images)
    with open(Path(output_filepath) / 'labels_pca.npy', 'wb') as f:
        np.save(f, all_labels)
    with open(Path(output_filepath) / 'test_images_pca.npy', 'wb') as f:
        np.save(f, test_images)
    with open(Path(output_filepath) / 'test_labels_pca.npy', 'wb') as f:
        np.save(f, test_labels)

    logger.info('Data processed and saved')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()
