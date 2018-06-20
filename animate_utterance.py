"""
Functions to create an animation of an utterance.

Date: Dec 2017
Author: Aciel Eshky

An utterance consists of a tuple of 4 files: a prompt file (.txt), an audio file (.wav), an ultrasound file (.ult) and
ultrasound parameter file (.param).

These functions use the ultrasound parameters to interpret and animate the ultrasound while synchronising it with the
audio and displaying the original prompt.
"""

# write a function to reduce ultrasound frame rate.
import os
import sys
import subprocess
import shutil
import matplotlib.pyplot as plt
from transform_ultrasound import transform_raw_ult_to_world_multi_frames
from reshape_ultrasound import reshape_ultrasound_array
from read_core_files import *
from reshape_ultrasound import reduce_frame_rate


def write_images_to_disk(ult_3d, directory, title=None):
    """
    A function to write the ultrasound frames as images to a directory. The images are generated as plots without axes.
    :param ult_3d: input ultrasound object as a 3d numpy array
    :param directory: the directory to write the images to
    :param title: an optional title for the image
    :return:
    """
    print("writing image frames to disk...")

    plt.figure(dpi=300)

    if title is not None:
        plt.title(title)

    c = ult_3d[0]
    im = plt.imshow(c.T, aspect='equal', origin='lower', cmap='gray')
    for i in range(1, ult_3d.shape[0]):
        c = ult_3d[i]
        im.set_data(c.T)
        plt.axis("off")
        plt.savefig(directory + "/%07d.jpg" % i, bbox_inches='tight')


def create_video(ult_3d, frame_rate, output_video_file, title=None):
    """
    A function to animate an ultrasound object.
    :param ult_3d: input ultrasound object as a 3d numpy array. Can be raw or transformed.
    :param frame_rate: taken from the ultrasound parameter file (unless the ultrasound has been down-sampled)
    :param output_video_file: the path/name of the output video
    :param title: an optional title for the video
    :return:
    """
    print("creating temporary directory...")
    directory = '.temp'
    if not os.path.exists(directory):
        os.makedirs(directory)

    write_images_to_disk(ult_3d=ult_3d, directory=directory, title=title)

    print("creating video from images frames using ffmpeg...")
    subprocess.call(
        ["ffmpeg", "-y", "-r", str(frame_rate),
         "-i", directory + "/%07d.jpg", "-vcodec", "mpeg4", "-qscale", "5", "-r",
         str(frame_rate), output_video_file])
    print("video saved.")

    shutil.rmtree(directory)
    print("image frames files deleted from disk.")


def crop_audio(audio_start_time, input_audio_file, output_audio_file):
    """
    A function to crop the audio. I should consider the cases where audio start time is 0 or negative
    :param audio_start_time: taken from the ultrasound parameter file: 'TimeInSecsOfFirstFrame'
    :param input_audio_file: path/name of input audio
    :param output_audio_file: path/name of output audio
    :return:
    """
    print("cropping audio...")

    subprocess.call(
        ["ffmpeg", "-ss", str(audio_start_time), "-i", input_audio_file, output_audio_file])


def append_audio_and_video(audio_file, video_file, output_video_file):
    """
    Outputs the video file with audio.
    :param audio_file:
    :param video_file:
    :param output_video_file:
    :return:
    """
    print("appending audio to video...")

    subprocess.call(
        ["ffmpeg",
         "-i", audio_file,
         "-i", video_file,
         "-codec", "copy", "-shortest", output_video_file])


def animate_utterance(prompt_file, wave_file, ult_file, param_file, output_video_file, frame_rate=60,
                      background_colour=0):
    """

    :param prompt_file:
    :param wave_file:
    :param ult_file:
    :param param_file:
    :param output_video_file:
    :param frame_rate:
    :param background_colour:
    :return:
    """

    # temp file names
    temp_audio_file = "cropped_audio.wav"
    temp_video_file = "video_only.avi"

    # prompt file is used for a video caption
    video_caption = ', '.join(parse_prompt_file(prompt_file))

    # read parameter file
    param_df = parse_parameter_file(param_file=param_file)

    # use offset parameter to crop audio
    crop_audio(audio_start_time=param_df['TimeInSecsOfFirstFrame'].value,
               input_audio_file=wave_file,
               output_audio_file=temp_audio_file)

    # read ultrasound, reshape it, reduce the frame rate for efficiency, and transform it
    ult = read_ultrasound_file(ult_file=ult_file)

    ult_3d = reshape_ultrasound_array(ult, output_dim=3,
                                      number_of_vectors=int(param_df['NumVectors'].value),
                                      pixels_per_vector=int(param_df['PixPerVector'].value))

    x, fps = reduce_frame_rate(ult_3d=ult_3d, input_frame_rate=float(param_df['FramesPerSec'].value),
                          output_frame_rate=frame_rate)

    print("transforming raw ultrasound to world...")
    y = transform_raw_ult_to_world_multi_frames(x, background_colour=background_colour)

    # create video without audio
    create_video(y, fps, temp_video_file, title=video_caption)

    # append audio and video
    append_audio_and_video(temp_audio_file, temp_video_file, output_video_file)

    # remove temporary files
    os.remove(temp_audio_file)
    os.remove(temp_video_file)

    print("video creation complete.")


def main():

    path = sys.argv[1]
    utterance_basename = sys.argv[2]
    output_path = sys.arg[3]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    prompt = os.path.join(path, utterance_basename + ".txt")
    wave = os.path.join(path, utterance_basename + ".wav")
    ultrasound = os.path.join(path, utterance_basename + ".ult")
    parameter = os.path.join(path, utterance_basename + ".param")

    animate_utterance(prompt, wave, ultrasound, parameter, os.path.join(output_path, "sample_video.avi"))


if __name__ == "__main__":
    main()
