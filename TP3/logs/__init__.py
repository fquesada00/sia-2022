from .files import TRAIN_ERROR_BY_EPOCH_FILE_PATH, TEST_ERROR_BY_EPOCH_FILE_PATH


def log_error_by_epoch(metrics_output_path, epochs, set_type: str):

    if set_type == 'train':
        file_path = TRAIN_ERROR_BY_EPOCH_FILE_PATH
    elif set_type == 'test':
        file_path = TEST_ERROR_BY_EPOCH_FILE_PATH
    else:
        return Exception(f"Invalid set type: {set_type}")

    with open(metrics_output_path) as f:
        for line_index, line in enumerate(f):
            if line_index == 5:
                with open(file_path, "a") as error_by_epoch_file:
                    error_by_epoch_file.write(
                        f"{epochs} {float(line.split()[0])}\n")
                break
