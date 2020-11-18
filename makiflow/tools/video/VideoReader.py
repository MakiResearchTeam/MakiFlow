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
        for _ in range(n):
            ret, frame = self._video.read()
            if not ret:
                break

            frames.append(transform(frame))

        assert len(frames) != 0, 'There are no frames left. Please, reset the video reader. (video is opened, for devs)'
        # Pad lacking frames
        if len(frames) != n:
            k = len(frames)
            to_add = n - k
            frames = frames + [frames[-1]] * to_add

        return frames, self._video.isOpened()

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

        frame_batch, has_frames = self.read_frames(batch_size)
        while has_frames:
            yield frame_batch
            frame_batch, has_frames = self.read_frames(batch_size)

        raise StopIteration('The video is read. Please, reset the video reader.')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-v', '--video', help='Path the video.')
    parser.add_argument('-b', '--batch_size', help='Which batch_size to test.', type=int)
    args = parser.parse_args()

    video_reader = VideoReader(args.video)
    print('FPS:', video_reader.get_fps())
    print('Frame size:', video_reader.get_frame_size())
    print('Video lenght:', video_reader.get_length())

    i = 0
    for frame_batch in video_reader.get_iterator(args.batch_size):
        assert len(frame_batch) == args.batch_size
        i += 1

    print('Number of iterations before reset:', i)

    video_reader.reset()

    i = 0
    for frame_batch in video_reader.get_iterator(args.batch_size, lambda x: x[:100, :100]):
        assert len(frame_batch) == args.batch_size
        assert frame_batch[0].shape[:2] == (100, 100)
        i += 1

    print('Number of iterations after reset:', i)
