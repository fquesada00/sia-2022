from file_read_backwards import FileReadBackwards

from .files import TRAIN_ERROR_BY_EPOCH_FILE_PATH

def log_train_error_by_epoch(metrics_output_path, epochs):
    with FileReadBackwards(metrics_output_path, encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if line_index == 1:
                with open(TRAIN_ERROR_BY_EPOCH_FILE_PATH, "a") as train_error_by_epoch_file:
                    train_error_by_epoch_file.write(f"{epochs} {float(line.split()[0])}\n")
                break