import torch

def scaled_CFG(difference_scales, steering_scale, base_scale, total_scale):
    def combine_predictions(predictions):
        base = base_scale(predictions[0])
        steering = base * 0
        len_predictions = len(predictions)
        for (a,b,scale) in difference_scales:
            if a >= len_predictions or b >= len_predictions: continue
            steering += scale(predictions[a] - predictions[b])
        return total_scale(base + steering_scale(steering))
    return combine_predictions

def apply_dynthresh(predictions_split, noise_prediction, target, percentile):
    target_prediction = predictions_split[1] + target * (predictions_split[1] - predictions_split[0])
    flattened_target = torch.flatten(target_prediction, 2)
    target_mean = flattened_target.mean(dim=2)
    for dim_index in range(flattened_target.shape[1]):
        flattened_target[:,dim_index] -= target_mean[:,dim_index]
    target_thresholds = torch.quantile(flattened_target.abs().float(), percentile, dim=2)
    flattened_prediction = torch.flatten(noise_prediction, 2)
    prediction_mean = flattened_prediction.mean(dim=2)
    for dim_index in range(flattened_prediction.shape[1]):
        flattened_prediction[:,dim_index] -= prediction_mean[:,dim_index]
    thresholds = torch.quantile(flattened_prediction.abs().float(), percentile, dim=2)
    for dim_index in range(noise_prediction.shape[1]):
        noise_prediction[:,dim_index] -= prediction_mean[:,dim_index]
        noise_prediction[:,dim_index] *= target_thresholds[:,dim_index] / thresholds[:,dim_index]
        noise_prediction[:,dim_index] += prediction_mean[:,dim_index]

def apply_naive_rescale(predictions_split, noise_prediction):
    get_scale = lambda p: torch.linalg.vector_norm(p, ord=2).item() / p.numel()
    norms = [get_scale(x) for x in predictions_split]
    natural_scale = sum(norms) / len(norms)
    final_scale = get_scale(noise_prediction)
    noise_prediction *= natural_scale / final_scale