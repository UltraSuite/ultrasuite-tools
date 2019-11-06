"""
Ultrasound utilities including:
 a function to reshape the ultrasound array into 1D, 2D or 3D
 a function to reduce the frame rate of a sequence of ultrasound frames
 a function to return a segment of ultrasound or audio by start and end times

Date: Mar 2018
Author: Aciel Eshky

"""
import numpy as np


def reduce_frame_rate(ult_3d, input_frame_rate=121.5, output_frame_rate=60):
    """
    Reduce the ultrasound frame rate to make other processes more efficient.
    :param ult_3d:
    :param input_frame_rate:
    :param output_frame_rate:
    :return:
    """
    print("reducing ultrasound frame rate from " + str(input_frame_rate) + " to " + str(output_frame_rate) + "...")

    if input_frame_rate < output_frame_rate:
        print("Output frame is larger than input frame. Frame rate not reduced.")
        return ult_3d

    skip = round(input_frame_rate / output_frame_rate)

    indices_of_selected_frames = range(0, ult_3d.shape[0], skip)

    y = np.empty([len(indices_of_selected_frames), ult_3d.shape[1], ult_3d.shape[2]])
    j = 0
    for i in indices_of_selected_frames:
        y[j] = ult_3d[i]
        j += 1

    return y, input_frame_rate / skip


