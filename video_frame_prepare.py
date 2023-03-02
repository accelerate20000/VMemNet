import ffmpeg
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--video-dir', default='/home/media/VideoMem/sources/')
parser.add_argument('--frame-dir', default='/home/media/VideoMem/frames')
parser.add_argument('--frame-num', type=int, default=24*7)
parser.add_argument('--frame-step', type=int, default=4)


def read_frame_as_jpeg(in_filename, frame_num):
    out, err = (
        ffmpeg
        .input(in_filename)
        .filter('select', 'gte(n,{})'.format(frame_num))
        .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
        .run(capture_stdout=True)
    )
    return out


if __name__ == '__main__':

    args = parser.parse_args()

    for root, ds, fs in os.walk(args.video_dir):
        for file in fs:
            fullname = os.path.join(root, file)
            video = fullname
            for frames in range(0, args.frame_num, args.frame_step):
                out = read_frame_as_jpeg(video, frames)
                frame_dir = os.path.join(args.frame_dir, file)
                frame_dir, _ = frame_dir.split('.webm')
                if not os.path.exists(frame_dir):
                    os.makedirs(frame_dir)
                with open(frame_dir + '/{}.jpg'.format(frames), 'wb') as f:
                    f.write(out)

