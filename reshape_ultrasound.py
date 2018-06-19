"""
Functions to transform the ultrasound from raw reflection data to real world proportions and produce images
of either the raw or transformed data.

Date: Mar 2018
Author: Aciel Eshky

"""
import sys
import numpy as np
from read_core_files import parse_parameter_file, read_ultrasound_file
from visualise_ultrasound import display_2d_ultrasound_frame


def reshape_ultrasound_array(ult, output_dim, number_of_vectors=63, pixels_per_vector=412):

    """
    A function to reshape a numpy array containing ultrasound data.
    The function can reshape a single ultrasound frame. Accepted dimensions: {2, 3}.
    The function can also reshape multiple ultrasound frames. Accepted dimensions: {1, 2, 3}.

    :param ult: ultrasound data as numpy array
    :param output_dim: 1, 2, or 3
    :param number_of_vectors: number of ultrasound scan lines, usually 63
    :param pixels_per_vector: number of datapoints per scanline, usually 412
    :return: the reshaped ultrasound data
    """

    valid_dim = {1, 2, 3}

    if output_dim not in valid_dim:
        raise ValueError("results: output_dim must be one of %r." % valid_dim)

    frame_size = number_of_vectors * pixels_per_vector  # this is usually 63 x 412 = 25,956

    number_of_frames = ult.size // frame_size # here i should check that this results in an integer.

    if number_of_frames == 1:  # we are dealing in this case with a single ultrasound frame

        if output_dim == 1:
            new_shape = (number_of_vectors * pixels_per_vector)

        elif output_dim == 2:
            new_shape = (number_of_vectors, pixels_per_vector)

        else:  # neither 1 nor 2: raise an error
            raise ValueError("Invalid output dimension provided for a single ultrasound frame. " +
                             "Expected one of [1, 2].")
    else:

        if output_dim == 1:
            new_shape = (number_of_frames * number_of_vectors * pixels_per_vector)

        elif output_dim == 2:
            new_shape = (number_of_frames, number_of_vectors * pixels_per_vector)

        elif output_dim == 3:
            new_shape = (number_of_frames, number_of_vectors, pixels_per_vector)

        else:  # not 1, 2, nor 3: raise an error
            raise ValueError("Invalid output dimension provided for a sequence of ultrasound frames. " +
                             "Expected one of [1, 2, 3].")

    return np.reshape(ult, new_shape)


def reduce_frame_rate(ult_3d, input_frame_rate=121.5, output_frame_rate=65):
    """
    Reduce the ultrasound frame rate to make other processes more efficient.
    :param ult_3d:
    :param input_frame_rate:
    :param output_frame_rate:
    :return:
    """
    print("reducing frame rate from " + str(input_frame_rate) + " to " + str(output_frame_rate) + "...")

    if input_frame_rate < output_frame_rate:
        print("Output frame is larger than input frame. Frame rate not reduced.")
        return ult_3d

    skip = input_frame_rate // output_frame_rate

    y = np.empty([ult_3d.shape[0] // skip, ult_3d.shape[1], ult_3d.shape[2]])
    j = 0
    for i in range(0, ult_3d.shape[0], skip):
        y[j] = ult_3d[i]
        j += 1

    return y


def get_segment(waveform, start_time, end_time=None, sampling_rate=22050):
    """
    A function to get part of the ultrasound or waveform, where start and end are specified as time
    :param waveform:
    :param start_time: in seconds
    :param end_time: in seconds
    :param sampling_rate: or frame rate
    :return:
    """
    start_frame = sampling_rate * start_time
    if end_time is None:
        return waveform[round(start_frame):]
    end_frame = sampling_rate * end_time
    return waveform[round(start_frame):round(end_frame)]


def main():

    ult_filename = sys.argv[1]      # "demo/utterance/Ultrax_02TD1M_001.ult"
    param_filename = sys.argv[2]    # "demo/utterance/Ultrax_02TD1M_001.param"

    ult = read_ultrasound_file(ult_filename)
    param_df = parse_parameter_file(param_filename)

    scanlines = int(param_df['NumVectors'].value)
    pixels_per_scanline = int(param_df['PixPerVector'].value)

    reshaped_ult = reshape_ultrasound_array(ult, output_dim=3, number_of_vectors=scanlines,
                                            pixels_per_vector=pixels_per_scanline)

    print("Displaying ultrasound file (check plotting window)...")
    display_2d_ultrasound_frame(reshaped_ult[100], title="ultrasound frame number 100")  # plot the 100th frame


if __name__ == "__main__":
    main()
