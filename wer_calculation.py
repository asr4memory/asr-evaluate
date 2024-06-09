from pathlib import Path
import os
import sys
from jiwer import process_words, process_characters, visualize_alignment
from app_config import get_config


def list_files(directory):
    "Function to get the base name of the files up to the first point."
    files = {}
    for filename in os.listdir(directory):
        # Split the file name at the first point.
        base_name = filename.split(".", 1)[0]
        files[base_name] = filename
    return files


def run_on_directories(print_alignment=False):
    # List files in both directories:
    config = get_config()["wer_calculation"]
    reference_directory = config["reference_directory"]
    hypothesis_directory = config["hypothesis_directory"]
    reference_files = list_files(reference_directory)
    hypothesis_files = list_files(hypothesis_directory)

    # Iterate through the reference files and check for matches in the
    # hypothesis files:
    for ref_base, ref_filename in reference_files.items():

        if ref_base not in hypothesis_files:
            # print(f"No corresponding hypothesis file found for: {ref_base}")
            continue

        if ref_filename == ".DS_Store":
            continue

        # Construct complete path to the files:
        ref_file_path = os.path.join(reference_directory, ref_filename)
        hyp_file_path = os.path.join(hypothesis_directory, hypothesis_files[ref_base])

        # Read texts from the files:
        with open(ref_file_path, "r", encoding="utf-8") as ref:
            reference_text = ref.read()
        with open(hyp_file_path, "r", encoding="utf-8") as hyp:
            hypothesis_text = hyp.read()

        # Calculate WER, MER and WIL:
        metrics = process_words(reference_text, hypothesis_text)
        char_output = process_characters(reference_text, hypothesis_text)

        print(
            f"For the file pair: {ref_base} => comparing reference file '{ref_filename}' vs. hypothesis file '{hypothesis_files[ref_base]}'"
        )
        print(f"WER: {metrics.wer}")
        print(f"MER: {metrics.mer}")
        print(f"WIL: {metrics.wil}")
        print(f"CER: {char_output.cer}")

        # Optional: Output alignments and visual representation of the
        # alignment:
        if print_alignment:
            print(visualize_alignment(char_output))


if __name__ == "__main__":
    if sys.argv[-1] == "-a" or sys.argv[-1] == "--print-alignment":
        run_on_directories(print_alignment=True)
    else:
        run_on_directories()
