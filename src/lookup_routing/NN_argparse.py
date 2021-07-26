import argparse


def _handle_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Set the number of epochs (e > 0)",
        dest="epochs",
        default=3,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Set the batch size (b > 0)",
        dest="batch_size",
        default=1000,
    )
    parser.add_argument(
        "-v",
        "--validation_samples",
        help="Set the validation sample size",
        dest="num_samp_val",
        default=100000,
    )
    parser.add_argument(
        "-p",
        "--prediction_samples",
        help="Set the prediction sample size",
        dest="num_samp_pred",
        default=1000,
    )
    parser.add_argument(
        "-a",
        "--array_length",
        help="Set the array length or number of slices between the min and max of each input parameter",
        dest="AL",
        default=4,
    )

    args = parser.parse_args()

    return args
