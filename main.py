#!/usr/bin/python3 -u


from dipy.data.fetcher import (fetch_bundle_atlas_hcp842,
                               get_bundle_atlas_hcp842)
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.segment.bundles import RecoBundles
from dipy.tracking.streamline import Streamlines
from time import time


import glob
import json
import logging
import numpy as np
import os


def get_hcp842_atlas_bundles():
    atlas_dict, atlas_folder = fetch_bundle_atlas_hcp842()
    atlas_file, all_bundles_files = get_bundle_atlas_hcp842()
    bundles_fnames = glob.glob(all_bundles_files)
    return bundles_fnames


if __name__ == '__main__':
    # Create Brainlife's output dirs if don't exist
    out_dir = 'output'
    if not os.path.exists('output'):
        os.mkdir(out_dir)

    # Read Brainlife's config.json
    with open('config.json', encoding='utf-8') as config_json:
        config = json.load(config_json)

    # TODO: Select atlas
    model_bundle_files = config.get('model_bundle_files')
    bundles_fnames = get_hcp842_atlas_bundles()

    streamline_files = config.get('streamline_files')

    greater_than = config.get('greater_than')
    less_than = config.get('less_than')
    no_slr = config.get('no_slr')
    clust_thr = config.get('clust_thr')
    reduction_thr = config.get('reduction_thr')
    reduction_distance = config.get('reduction_distance')
    model_clust_thr = config.get('model_clust_thr')
    pruning_thr = config.get('pruning_thr')
    pruning_distance = config.get('pruning_distance')
    slr_metric = config.get('slr_metric')
    slr_transform = config.get('slr_transform')
    slr_matrix = config.get('slr_matrix')
    refine = config.get('refine')
    r_reduction_thr = config.get('r_reduction_thr')
    r_pruning_thr = config.get('r_pruning_thr')
    no_r_slr = config.get('no_r_slr')

    slr = not no_slr
    r_slr = not no_r_slr

    bounds = [(-30, 30), (-30, 30), (-30, 30),
              (-45, 45), (-45, 45), (-45, 45),
              (0.8, 1.2), (0.8, 1.2), (0.8, 1.2)]

    slr_matrix = slr_matrix.lower()
    if slr_matrix == 'nano':
        slr_select = (100, 100)
    if slr_matrix == 'tiny':
        slr_select = (250, 250)
    if slr_matrix == 'small':
        slr_select = (400, 400)
    if slr_matrix == 'medium':
        slr_select = (600, 600)
    if slr_matrix == 'large':
        slr_select = (800, 800)
    if slr_matrix == 'huge':
        slr_select = (1200, 1200)

    slr_transform = slr_transform.lower()
    if slr_transform == 'translation':
        bounds = bounds[:3]
    if slr_transform == 'rigid':
        bounds = bounds[:6]
    if slr_transform == 'similarity':
        bounds = bounds[:7]
    if slr_transform == 'scaling':
        bounds = bounds[:9]

    logging.info('### RecoBundles ###')

    t = time()
    logging.info(streamline_files)
    input_obj = load_tractogram(streamline_files, 'same',
                                bbox_valid_check=False)
    streamlines = input_obj.streamlines

    logging.info(' Loading time %0.3f sec' % (time() - t,))

    rb = RecoBundles(streamlines, greater_than=greater_than,
                     less_than=less_than, clust_thr=clust_thr)

    bcode = []
    blabel = []
    names = []
    sls = Streamlines()
    for ind, mb in enumerate(bundles_fnames[:3]):
        model_bundle = load_tractogram(mb, 'same',
                                       bbox_valid_check=False).streamlines

        recognized_bundle, labels = rb.recognize(
            model_bundle, model_clust_thr=model_clust_thr,
            reduction_thr=reduction_thr, reduction_distance=reduction_distance,
            pruning_thr=pruning_thr, pruning_distance=pruning_distance,
            slr=slr, slr_metric=slr_metric, slr_x0=slr_transform,
            slr_bounds=bounds, slr_select=slr_select, slr_method='L-BFGS-B')

        sls.extend(recognized_bundle)

        length = len(labels)
        print(length)

        bcode.extend([ind] * length)
        if length > 0:
            blabel.extend(labels.tolist())
            name = os.path.basename(mb).split('.')[0]
            names.append(name)

    data_per_streamline = {'bundle_code': bcode, 'bundle_label': blabel}

    sft = StatefulTractogram(sls, streamline_files, Space.RASMM,
                             data_per_streamline=data_per_streamline)
    save_tractogram(sft, os.path.join(out_dir, 'track.trk'),
                    bbox_valid_check=False)

    # TODO: The header cannot be modified. Bundle names should be saved in
    #  a txt file

    bcode = []
    bname = []
    blabel = []
    sls = Streamlines()
    for ind, mb in enumerate(bundles_fnames[:3]):
        t = time()
        logging.info(mb)
        model_bundle = load_tractogram(mb, 'same',
                                       bbox_valid_check=False).streamlines
        logging.info(' Loading time %0.3f sec' % (time() - t,))
        logging.info("model file = ")
        logging.info(mb)

        recognized_bundle, labels = rb.recognize(
            model_bundle, model_clust_thr=model_clust_thr,
            reduction_thr=reduction_thr, reduction_distance=reduction_distance,
            pruning_thr=pruning_thr, pruning_distance=pruning_distance,
            slr=slr, slr_metric=slr_metric, slr_x0=slr_transform,
            slr_bounds=bounds, slr_select=slr_select, slr_method='L-BFGS-B')

        if refine:
            if len(recognized_bundle) > 1:
                # affine
                x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])
                affine_bounds = [(-30, 30), (-30, 30), (-30, 30),
                                 (-45, 45), (-45, 45), (-45, 45),
                                 (0.8, 1.2), (0.8, 1.2), (0.8, 1.2),
                                 (-10, 10), (-10, 10), (-10, 10)]

                recognized_bundle, labels = rb.refine(
                    model_bundle, recognized_bundle,
                    model_clust_thr=model_clust_thr,
                    reduction_thr=r_reduction_thr,
                    reduction_distance=reduction_distance,
                    pruning_thr=r_pruning_thr,
                    pruning_distance=pruning_distance, slr=r_slr,
                    slr_metric=slr_metric, slr_x0=x0, slr_bounds=affine_bounds,
                    slr_select=slr_select, slr_method='L-BFGS-B')

        if len(labels) > 0:
            ba, bmd = rb.evaluate_results(model_bundle, recognized_bundle,
                                          slr_select)

            logging.info("Bundle adjacency Metric {0}".format(ba))
            logging.info("Bundle Min Distance Metric {0}".format(bmd))
