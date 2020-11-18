import cv2


class VideoReader:
    def __init__(self, video_path):
        """
        A utility for batching frames from a video file.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        """
        self._path = video_path
        self._video = None
        self._last_frame = None
        self.reset()

    def reset(self):
        """
        Resets the state of the reader, so that it can read frames again.

        Returns
        -------

        """
        if self._video is not None:
            self._video.release()

        self._video = cv2.VideoCapture(self._path)
        assert self._video.isOpened(), f'Could not open video with path={self._path}'
        self._last_frame = None

    def get_length(self):
        """
        Returns
        -------
        int
            Number of frames in the video.
        """
        return int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_size(self):
        """
        Returns
        -------
        (int, int)
            Frame height and width.
        """
        height = int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))
        return height, width

    def get_fps(self):
        """
        Returns
        -------
        int
            Fps number.
        """
        return self._video.get(cv2.CAP_PROP_FPS)

    def read_frames(self, n, transform=None) -> (list, bool):
        """
        Reads a batch of frames and returns them packed in list.
        If there are not enough enough frames for the batch,
        it will pad the missing frames with the last one and also return False in the flag.

        Parameters
        ----------
        n : int
            How many frames to read.
        transform : python function
            Will be applied to each frame.

        Returns
        -------
        list
            List of n frames.
        bool
            Flag that shows if there are frames left in the video.
        """
        assert self._video.isOpened(), 'There are no frames left. Please, reset the video reader.'

        if transform is None:
            transform = lambda x: x

        frames = []
        # Transform and add to the list the last frame if it is present
        if self._last_frame is not None:
            frame = transform(self._last_frame)
            frames.append(frame)
            self._last_frame = None

        for _ in range(n - len(frames)):
            ret, frame = self._video.read()
            if not ret:
                print('Ran out of frames.')
                break

            frames.append(transform(frame))

        assert len(frames) != 0, 'There are no frames left. Please, reset the video reader. (video is opened, for devs)'

        # Pad lacking frames
        if len(frames) != n:
            k = len(frames)
            to_add = n - k
            frames = frames + [frames[-1]] * to_add
        # Sanity check
        assert len(frames) == n, f'Number of frames={len(frames)} is not equal to the requested amount={n}'

        # This is used to check whether there are frames left.
        ret, frame = self._video.read()
        self._last_frame = frame

        return frames, ret

    def get_iterator(self, batch_size, transform=None):
        """
        Creates an iterator that yields batches of frames from the video.

        Parameters
        ----------
        batch_size : int
            The batch size.
        transform : python function
            Will be applied to each frame in the batch.

        Returns
        -------
        python iterator
        """

        frame_batch, has_frames = self.read_frames(batch_size, transform=transform)
        while has_frames:
            yield frame_batch
            frame_batch, has_frames = self.read_frames(batch_size, transform=transform)

        raise StopIteration('The video is read. Please, reset the video reader.')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-v', '--video', help='Path the video.')
    parser.add_argument('-b', '--batch_size', help='Which batch_size to test.', type=int, default=1)
    args = parser.parse_args()

    print('Path:', args.video)
    print('Batch size:', args.batch_size)

    video_reader = VideoReader(args.video)
    print('FPS:', video_reader.get_fps())
    print('Frame size:', video_reader.get_frame_size())
    print('Video lenght:', video_reader.get_length())

    i = 0
    for frame_batch in video_reader.get_iterator(args.batch_size):
        assert len(frame_batch) == args.batch_size, f'Expected batch_size={args.batch_size}, received {len(frame_batch)}'
        i += 1

    print('Number of iterations before reset:', i)

    video_reader.reset()

    i = 0
    for frame_batch in video_reader.get_iterator(args.batch_size, lambda x: x[:100, :100]):
        assert len(frame_batch) == args.batch_size
        assert frame_batch[0].shape[:2] == (100, 100), f'Expected shape {(100, 100)}, got {frame_batch[0].shape[:2]}, iteration={i}'
        i += 1

    print('Number of iterations after reset:', i)
    from makiflow.core.debug_utils import DebugContext

    with DebugContext('Check pure frame reading.'):
        video_reader.reset()
        frames, ret = video_reader.read_frames(args.batch_size)
        while ret:
            frames, ret = video_reader.read_frames(args.batch_size)
        print('Finished reading.')
        frames, ret = video_reader.read_frames(args.batch_size)
