import argparse
import collections
import os
import numpy as np
import torch
from parse_config import ConfigParser
from utils.util import get_feature_names
from shap_utils import (plot_beeswarm, plot_beeswarm_grouped,
                        plot_grouped_feature_importance, plot_shap_temporal_heatmap, plot_shap_difference_bar,
                        plot_shap_difference_aggregated, plot_shap_waterfall, plot_shap_waterfall_grouped,
                        map_sample_ids_to_indices,
                        compute_physical_consistency_score, compute_grouped_physical_consistency_score,
                        plot_beeswarm_by_grouped_feature, plot_beeswarm_by_feature)

def load_shap_inputs_from_combined_npz(shap_path, model_id, model_type):
    combined_npz_path = os.path.join(shap_path, f"shap_values_{model_id}_{model_type}_combined.npz")
    data = np.load(combined_npz_path, allow_pickle=True)

    shap_values = {
        "class_0": data["class_0"],
        "class_1": data["class_1"]
    }
    labels = data["label"]
    sample_ids = data["sample_id"]

    return shap_values, labels, sample_ids

def main(config):
    logger = config.get_logger('shap')

    checkpoint_path = config["XAI"]["checkpoint_path"]
    model_type = config["model_type"]
    only_pos = config["XAI"]["only_positive"]
    only_neg = config["XAI"]["only_negative"]
    shap_path = config["shap"]["shap_path"]
    all_model_path = '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/all_model_comparison'
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    feature_names = get_feature_names(config)

    shap_data, labels, sample_ids = load_shap_inputs_from_combined_npz(shap_path, model_id, model_type)

    shap_class = config["shap"]["class"]
    shap_values = shap_data[f"class_{shap_class}"]
    print(f"SHAP shape: {shap_values.shape}, Labels shape: {labels.shape}, Unique labels: {np.unique(labels)}")

    input_tensor_path = os.path.join(shap_path, f"shap_values_{model_id}_{model_type}_input.npy")
    input_tensor_np = np.load(input_tensor_path)
    input_tensor = torch.tensor(input_tensor_np)

    if only_pos:
        mask = labels == 1
        shap_values = shap_values[mask]
        input_tensor = input_tensor[mask]
        sample_ids = sample_ids[mask]
        labels = labels[mask]
        model_id += "_positive"
        print("Only positive SHAP values selected:", shap_values.shape)
    elif only_neg:
        mask = labels == 0
        shap_values = shap_values[mask]
        input_tensor = input_tensor[mask]
        sample_ids = sample_ids[mask]
        labels = labels[mask]
        model_id += "_negative"
        print("Only negative SHAP values selected:", shap_values.shape)
    else:
        print("Using all SHAP values:", shap_values.shape)

    shap_files = [
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/cnn/0615_020509/shap_values_0517_181322_cnn.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/mlp/0615_020230/shap_values_0517_175347_mlp.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/gru/0616_141854/shap_values_0514_140125_gru.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/lstm/0615_020730/shap_values_0513_230004_lstm.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/transformer/0615_025441/shap_values_0519_125059_transformer.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/gtn/0624_142112/shap_values_0623_205004_gtn.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/rf/0616_091841/shap_values_0612_082906_rf.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/tft/0616_133748/shap_values_0612_083316_tft.npz'
    ]

    input_files = [
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/cnn/0615_020509/shap_values_0517_181322_cnn_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/mlp/0615_020230/shap_values_0517_175347_mlp_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/gru/0616_141854/shap_values_0514_140125_gru_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/lstm/0615_020730/shap_values_0513_230004_lstm_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/transformer/0615_025441/shap_values_0519_125059_transformer_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/gtn/0624_142112/shap_values_0623_205004_gtn_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/rf/0616_091841/shap_values_0612_082906_rf_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/tft/0616_133748/shap_values_0612_083316_tft_input.npy'
    ]

    model_names = ['cnn', 'mlp', 'gru', 'lstm', 'transformer', 'gtn', 'rf', 'tft']
    features = ['lst_day_t-1', 'lst_day_t-2', 'rh_t-1', 't2m_t-1', 'd2m_t-1', 'lst_night_t-1', 'ndvi_t-1', 't2m_t-2', 'tp_t-1', 'wind_speed_t-1', 'lai_t-1', 'lst_day_t-5']
    grouped_features = [
        "d2m", "lai", "lst_day", "lst_night", "ndvi", "rh", "smi", "sp", "ssrd",
        "t2m", "tp", "wind_speed",
        "dem", "roads_distance", "slope", "lc_agriculture", "lc_forest", "lc_grassland",
        "lc_settlement", "lc_shrubland", "lc_sparse_vegetation", "lc_water_bodies",
        "lc_wetland", "population"
    ]

    features_to_plot = [
        "d2m_t-1", "lai_t-1",  "lst_day_t-1", "lst_night_t-1", "ndvi_t-1", "rh_t-1", "smi_t-1", "sp_t-1", "ssrd_t-1",
        "t2m_t-1", "tp_t-1",  "wind_speed_t-1",
        "dem_t-1",  "population_t-1",  "roads_distance_t-1", "slope_t-1", "lc_agriculture_t-1", "lc_forest_t-1",
        "lc_grassland_t-1", "lc_settlement_t-1",  "lc_shrubland_t-1",  "lc_sparse_vegetation_t-1",
        "lc_water_bodies_t-1", "lc_wetland_t-1"]

    #for feature in grouped_features:
        #plot_beeswarm_by_grouped_feature(shap_files=shap_files, input_files=input_files, feature_names=feature_names, feature_to_plot=feature, model_names=model_names, base_path=all_model_path, only_pos=only_pos, only_neg=only_neg)
       # plot_beeswarm_by_feature(shap_files=shap_files, input_files=input_files, feature_names=feature_names, full_feature_name=f"{feature}_t-1", model_names=model_names, base_path=all_model_path, only_pos=only_pos, only_neg=only_neg)

    #for idx in range(1, 30):
     #   plot_beeswarm_by_feature(shap_files=shap_files, input_files=input_files, feature_names=feature_names, full_feature_name=f"lai_t-{idx}", model_names=model_names, base_path=all_model_path, only_pos=only_pos, only_neg=only_neg)


    print(f"Shape input: {input_tensor.shape}, SHAP: {np.array(shap_values).shape}")
    plot_grouped_feature_importance(shap_values, feature_names, shap_path, model_type)
    #plot_beeswarm(shap_values, shap_class, input_tensor, feature_names, model_id, shap_path, model_type, logger)
    #plot_beeswarm_grouped(shap_values, shap_class, input_tensor, feature_names, model_id, shap_path, model_type, logger)

    #plot_shap_difference_bar(shap_data['class_0'], shap_data['class_1'], feature_names, model_id, shap_path, model_type, logger)
    #plot_shap_difference_aggregated(shap_data['class_0'], shap_data['class_1'], feature_names, model_id, shap_path, model_type, logger)

    #plot_shap_temporal_heatmap(shap_values, shap_class, feature_names, model_id, shap_path, model_type, logger)

    bigFire_sample_idx = [4792, 679, 8418, 1645, 1676]

    #Lists created in compare_models.ipynb
    false_positive_sample_ids = [
        16233, 15538, 15330, 15494, 16849, 15341, 17076, 16985, 16861, 15569,
        16701, 16838, 16896, 15759, 15687, 15391, 15593, 16919, 17079, 15647,
        14748, 15306, 14825, 17031, 16696, 16781, 16917, 16829, 16787, 15639,
        15767, 17000, 15624, 17028, 15499, 15559, 15621, 16823, 15812, 15459,
        15390, 15406, 17050, 16735, 15659, 15334, 15600, 15603, 16736, 15557,
        15452, 16789, 15356, 16996, 15382, 17226, 15397, 17038, 15758, 16688,
        17179, 17024, 15577, 17040, 17191, 15318, 15420, 15656, 15376, 15617,
        16949, 17139, 17001, 16886, 16687, 15503, 15466, 16713, 17007, 15398,
        16577, 15508, 15380, 15229, 16802, 15530, 16856, 15502, 15319, 16983,
        16940, 16938, 16810, 15324, 17059, 17215, 16754, 15400, 16897, 15572
    ]

    false_negative_sample_ids = [
        8449, 3713, 625, 4152, 4892, 6519, 8453, 8523, 6692, 6079, 3126, 671, 8448, 3131, 2952, 4801,
        4835, 4894, 4433, 4428, 5977, 6040, 3908, 702, 4800, 4832, 3920, 670, 3901, 4798, 3933, 6682,
        719, 8374, 1608, 661, 3080, 630, 3651, 3932, 6201, 3730, 3904, 2970, 701, 622, 1619, 1656, 6699,
        4422, 4836, 1639, 643, 3650, 4888, 6582, 8475, 5971, 2957, 8344, 6200, 6694, 1645, 665, 4893,
        3711, 6701, 5978, 2959, 4839, 6210, 5962, 632, 6121, 6067, 707, 3708, 3190, 3738, 6090, 4459,
        658, 3911, 1634, 5956, 3224, 2976, 3917, 623, 6522, 1642, 4927, 3912, 4806, 2962, 640, 1610,
        631, 8561, 8562
    ]

    true_negative_ids = [
        16608, 17285, 16028, 15754, 16217, 15344, 16648, 16618, 16099, 16241, 15160, 16087, 15028, 16201, 15865,
        15765, 15904, 16305, 17216, 14981, 16314, 16268, 16279, 16182, 15789, 14905, 15588, 14908, 16172, 14751,
        16353, 14821, 15796, 16549, 16877, 16370, 15438, 15888, 16527, 16137, 16599, 15187, 15004, 15186, 15488,
        14865, 16256, 15596, 16164, 15029, 15987, 16134, 16301, 16976, 16967, 15473, 17301, 14630, 14661, 16254,
        15629, 14940, 15299, 16360, 16073, 15157, 16051, 14965, 16436, 16402, 15689, 15684, 15923, 16517, 15001,
        14947, 15152, 17128, 14754, 16723, 15840, 16126, 17254, 14857, 16694, 16068, 16385, 15594, 15897, 16364,
        15755, 15084, 15925, 14814, 15154, 14848, 16281, 15202, 14985, 15704
    ]

    true_negative_ids_july = [
        16832, 16846, 16858, 16923, 16928, 16936
    ]


    for idx in bigFire_sample_idx:
        print(f"Plotting SHAP waterfall for Sample: {idx}")
        #plot_shap_waterfall_grouped(shap_values, shap_class, input_tensor, feature_names, sample_ids, idx, model_id, f"{shap_path}/big_fire_waterfall", model_type, logger)

    for idx in false_positive_sample_ids:
        print(f"Plotting SHAP waterfall for False Positive Sample: {idx}")
        #plot_shap_waterfall_grouped(shap_values, shap_class, input_tensor, feature_names, sample_ids, idx, model_id, f"{shap_path}/false_positive_waterfall_grouped", model_type, logger)

    for idx in true_negative_ids_july:
        print(f"Plotting SHAP waterfall for True Negative Sample: {idx}")
        plot_shap_waterfall_grouped(shap_values, shap_class, input_tensor, feature_names, sample_ids, idx, model_id, f"{shap_path}/true_negatives_waterfall_grouped", model_type, logger)


    #for idx in true_negative_ids:
    #    print(f"Plotting SHAP waterfall for Sample: {idx}")
        #plot_shap_waterfall_grouped(shap_values, shap_class, input_tensor, feature_names, sample_ids, idx, model_id, f"{shap_path}/true_negative_waterfall_grouped", model_type, logger)


    physical_knowledge = {
        "t2m": "+", "d2m": "-", "lc_agriculture": "+", "lc_forest": "+", "lc_grassland": "+",
        "lc_settlement": "-", "lc_shrubland": "+", "lc_sparse_vegetation": "+", "lc_water_bodies": "-",
        "lc_wetland": "-", "lst_day": "+", "lst_night": "+", "rh": "-", "roads_distance": "+", "slope": "+",
        "smi": "-", "ssrd": "+", "tp": "-", "wind_speed": "+"
    }


    #compute_grouped_physical_consistency_score( shap_values=shap_values, input_tensor=input_tensor,
           # feature_names=feature_names, physical_signs=physical_knowledge, save_path=shap_path, model_type=model_type)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Compute SHAP values')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to trained checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target nargs')
    options = [
        CustomArgs(['--cl', '--class'], type=int, target='shap;class', nargs=None),
        CustomArgs(['--only_positive', '--op'], type=lambda x: x.lower() in ['true', '1', 'yes'], target='XAI;only_positive', nargs=None),
        CustomArgs(['--only_negative', '--on'], type=lambda x: x.lower() in ['true', '1', 'yes'], target='XAI;only_negative', nargs=None),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
