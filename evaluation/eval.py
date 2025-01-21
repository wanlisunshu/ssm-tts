import argparse
import concurrent.futures
import json
import multiprocessing
import os
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import torch
# import utmosv2
from discrete_speech_metrics import MCD, LogF0RMSE, SpeechBERTScore
from torchaudio.pipelines import SQUIM_OBJECTIVE
from tqdm import tqdm

SR = 22050
UTMOSV2_CHECKPOINT = "https://github.com/sarulab-speech/UTMOSv2/raw/refs/heads/main/models/fusion_stage3/fold0_s42_best_model.pth"
UTMOSV2_FILE = "tmpdir_utmosv2/fold0_s42_best_model.pth"

if not os.path.exists(UTMOSV2_FILE):
    print(f"UTMOSv2 not found, downloading to {UTMOSV2_FILE}")
    os.makedirs("tmpdir_utmosv2")
    urllib.request.urlretrieve(UTMOSV2_CHECKPOINT, UTMOSV2_FILE)


speechbert = SpeechBERTScore(
    sr=SR,
    model_type="wavlm-large",
    layer=14,
    use_gpu=True,
)
mcd = MCD(sr=SR)
logf0 = LogF0RMSE(sr=SR)
# utmos = utmosv2.create_model(pretrained=True, checkpoint_path=UTMOSV2_FILE)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cuda:0')
print('Using device: ', device)

squim_obj = SQUIM_OBJECTIVE.get_model().to(device)


@torch.no_grad
def compute_metrics(ref, gen):
    sb_score, _, _ = speechbert.score(ref, gen)
    gen_ten = torch.tensor(gen)[None, :].to(device)
    stoi, pesq, si_sdr = squim_obj(gen_ten)
    return {
        "SpeechBERTScore": sb_score,
        "STOI": stoi.cpu().item(),
        "PESQ": pesq.cpu().item(),
        "SISDR": si_sdr.cpu().item(),
    }


def worker(input_data):
    # Each worker computes metrics for one input pair
    file, ref, gen = input_data
    mcd_score = mcd.score(ref, gen)
    logf0_score = logf0.score(ref, gen)
    print(f"{file}: {mcd_score} MCD; {logf0_score} LogF0")
    return file, mcd_score, logf0_score


def compute_metrics_parallel(inputs):
    """
    Parallelizes the computation of mcd.score and logf0.score over all available CPU cores.

    Args:
        inputs (list): A list of inputs, where each input is a tuple (audio1, audio2, etc.).
        mcd_func (function): The function to calculate the MCD score.
        logf0_func (function): The function to calculate the log-F0 score.

    Returns:
        list: A list of results, where each result is a tuple (mcd_score, logf0_score).
    """

    # Use the maximum number of available cores
    num_cores = multiprocessing.cpu_count()

    # Create a process pool and map the worker function to inputs
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_cores
    ) as executor:
        results = list(executor.map(worker, inputs))

    return results


def evaluate(ref_result_folder, gen_result_folder, split):

    results = {}
    ref_sample_folder = Path(ref_result_folder)
    gen_sample_folder = Path(gen_result_folder)

    mp_inputs = []

    print("Computing non-intrusive metrics")
    for file in tqdm(os.listdir(gen_sample_folder)):
        file_id = file.split("/")[-1].split("_")[0].replace(".wav", "")
        ref, _ = librosa.load(ref_sample_folder / f"{file_id}.wav")
        gen, _ = librosa.load(gen_sample_folder / f"{file_id}_generated.wav")

        results[file_id] = compute_metrics(ref, gen)
        mp_inputs.append((file_id, ref, gen))
    #
    # utmos_results = utmos.predict(input_dir=gen_sample_folder)
    #
    # for utmos_result in utmos_results:
    #     file_id = utmos_result["file_path"].split("/")[-1].replace(".wav", "")
    #     results[file_id]["UTMOSv2"] = utmos_result["predicted_mos"]

    print("Computing MCD and LogF0RMSE")
    mp_outputs = compute_metrics_parallel(mp_inputs)

    for file_id, mcd_score, logf0_score in mp_outputs:
        results[file_id]["MCD"] = mcd_score
        results[file_id]["LogF0RMSE"] = logf0_score

    json_file = Path(gen_result_folder) / f"eval_{split}.json"
    with open(json_file, "w") as f:
        json.dump(results, f)

    print("Computing statistics")
    aggregated_metrics = {}

    for file_id, metrics in tqdm(results.items()):
        for metric_name, value in metrics.items():
            if metric_name not in aggregated_metrics:
                aggregated_metrics[metric_name] = []
            aggregated_metrics[metric_name].append(value)

    statistics_results = {}
    for metric_name, values in tqdm(aggregated_metrics.items()):
        values_array = np.array(values)
        statistics_results[metric_name] = {
            "mean": float(np.mean(values_array)),
            "max": float(np.max(values_array)),
            "min": float(np.min(values_array)),
            "std": float(np.std(values_array)),
        }

    print("saving")
    json_file = Path(gen_result_folder) / f"eval_stats_{split}.json"
    with open(json_file, "w") as f:
        json.dump(statistics_results, f)
    print("done")


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        args (Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process reference and generated folders for a specific split."
    )
    parser.add_argument(
        "--ref_folder",
        type=str,
        required=True,
        help="Path to the reference folder containing the ground truth data.",
    )
    parser.add_argument(
        "--gen_folder",
        type=str,
        required=True,
        help="Path to the generated folder containing the output data.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        nargs="?",
        help="The data split to process (e.g., train, validation, test).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    evaluate(args.ref_folder, args.gen_folder, args.split)
