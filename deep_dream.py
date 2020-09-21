import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import tqdm
import scipy.ndimage as nd
from utils import deprocess, preprocess, clip
from models import model_selection
import cv2 as cv

import subprocess
import shutil
import re


def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


#Video Functions

def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img

def save_and_maybe_display_image(dump_dir, dump_img, should_display=False, name_modifier=None):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array got {type(dump_img)}.'

    # step1: figure out the dump dir location
    os.makedirs(dump_dir, exist_ok=True)

    # step2: define the output image name
    if name_modifier is not None:
        dump_img_name = str(name_modifier).zfill(6) + '.jpg'

    if dump_img.dtype != np.uint8:
        dump_img = (dump_img*255).astype(np.uint8)

    # step3: write image to the file system
    cv.imwrite(os.path.join(dump_dir, dump_img_name), dump_img[:, :, ::-1])  # ::-1 because opencv expects BGR (and not RGB) format...


# Return frame names that follow the 6 digit pattern and have .jpg extension
def valid_frames(input_dir):
    def valid_frame_name(str):
        pattern = re.compile(r'[0-9]{6}\.jpg')  # regex, examples it covers: 000000.jpg or 923492.jpg, etc.
        return re.fullmatch(pattern, str) is not None
    candidate_frames = sorted(os.listdir(input_dir))
    valid_frames = list(filter(valid_frame_name, candidate_frames))
    return valid_frames

def dump_frames(video_path, dump_dir):
    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg):  # if ffmpeg is in system path
        cap = cv.VideoCapture(video_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))

        input_options = ['-i', video_path]
        extract_options = ['-r', str(fps)]
        out_frame_pattern = os.path.join(dump_dir, 'frame_%6d.jpg')

        subprocess.call([ffmpeg, *input_options, *extract_options, out_frame_pattern])

        print(f'Dumped frames to {dump_dir}.')
        metadata = {'pattern': out_frame_pattern, 'fps': fps}
        return metadata
    else:
        raise Exception(f'{ffmpeg} not found in the system path, aborting.')

def create_video_from_intermediate_results(tmp_out, final_out, metadata=None):
    # save_and_maybe_display_image uses this same format (it's hardcoded there), not adaptive but does the job
    img_pattern = os.path.join(tmp_out, '%6d.jpg')
    fps = 5 if metadata is None else metadata['fps']
    first_frame = 0
    number_of_frames_to_process = len(valid_frames(tmp_out))  # default - don't trim process every frame
    out_file_name = 'deep_dream.mp4'

    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg):  # if ffmpeg is in system path
        input_options = ['-r', str(fps), '-i', img_pattern]
        trim_video_command = ['-start_number', str(first_frame), '-vframes', str(number_of_frames_to_process)]
        encoding_options = ['-c:v', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p']
        pad_options = ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']  # libx264 won't work for odd dimensions
        out_video_path = os.path.join(final_out, out_file_name)
        subprocess.call([ffmpeg, *input_options, *trim_video_command, *encoding_options, *pad_options, out_video_path])
        print(f'Saved video to {out_video_path}.')
        return out_video_path
    else:
        raise Exception(f'{ffmpeg} not found in the system path, aborting.')

def linear_blend(img1, img2, alpha=0.5):
    return img1 + alpha * (img2 - img1)

def deep_dream_video(vid_path, out, model, iterations, lr, octave_scale, num_octaves, max_frame, blend= 0.85):
    print('Max Frame:{}'.format(max_frame))
    #video_path = os.path.join(vid_path, config['input'])
    tmp_input_dir = os.path.join(out, 'tmp_input')
    tmp_output_dir = os.path.join(out, 'tmp_out')
    final_output_dir= os.path.join(out, 'final_out')
    os.makedirs(tmp_input_dir, exist_ok=True)
    os.makedirs(tmp_output_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    metadata = dump_frames(vid_path, tmp_input_dir)

    last_img = None
    for frame_id, frame_name in enumerate(sorted(os.listdir(tmp_input_dir))):
        print(f'Processing frame {frame_id}')
        frame_path = os.path.join(tmp_input_dir, frame_name)
        #modify to resize
        frame = load_image(frame_path)
        if blend is not None and last_img is not None:
            # 1.0 - get only the current frame, 0.5 - combine with last dreamed frame and stabilize the video
            frame = linear_blend(last_img, frame, blend)

        dreamed_frame = deep_dream(frame, model, iterations, lr, octave_scale, num_octaves)
        last_img = dreamed_frame
        save_and_maybe_display_image(tmp_output_dir, dreamed_frame, name_modifier=frame_id)
        if frame_id == max_frame:
            break

    create_video_from_intermediate_results(tmp_output_dir, final_output_dir, metadata)

    shutil.rmtree(tmp_input_dir)  # remove tmp files
    print(f'Deleted tmp frame dump directory {tmp_input_dir}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default="images/supermarket.jpg", help="path to input image")
    parser.add_argument("--iterations", type = int,  default=20, help="number of gradient ascent steps per octave")
    parser.add_argument("--at_layer",  type = int, default=22, help="layer at which we modify image to maximize outputs")
    parser.add_argument("--lr", type = float, default=0.01, help="learning rate")
    parser.add_argument("--octave_scale", type = float, default=1.4, help="image scale between octaves")
    parser.add_argument("--num_octaves", type = int, default=10, help="number of octaves")
    parser.add_argument("--load_checkpoint")
    parser.add_argument("--vid", type= bool, default = False)
    parser.add_argument("--input_vid", type=str, help="path to input video")
    parser.add_argument("--out", type=str, help="output path")
    parser.add_argument("--max_frame", type=str, help="maximum frame for video")

    args = parser.parse_args()


    # Define the model
    #model instantiation
    network, *_ = model_selection(modelname='xception', num_out_classes=2)
    checkpoint = torch.load(args.load_checkpoint)
    network.load_state_dict(checkpoint)
    #layers = list(network.model.children())
    #model = nn.Sequential(*layers[: (args.at_layer + 1)])
    model = network
    if torch.cuda.is_available:
        model = model.cuda()
  
    if args.vid:
        deep_dream_video(args.input_vid,
        args.out, 
        model, 
        args.iterations, 
        args.lr,
        args.octave_scale, 
        args.num_octaves,
        args.max_frame)

        exit()

    # Load image
    image = Image.open(args.input_image)
    # Extract deep dream image
    dreamed_image = deep_dream(
        image,
        model,
        iterations=args.iterations,
        lr=args.lr,
        octave_scale=args.octave_scale,
        num_octaves=args.num_octaves,
    )

    # Save and plot image
    os.makedirs("outputs", exist_ok=True)
    filename = args.input_image.split("/")[-1]
    #cv2.imwrite(f'outputs/{filename}', dreamed_image)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(dreamed_image)
    plt.imsave(f"outputs/output_{filename}", dreamed_image)
    plt.show()