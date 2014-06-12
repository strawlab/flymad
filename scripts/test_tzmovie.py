import subprocess
import tempfile

import roslib; roslib.load_manifest('flymad')

import flymad.conv
from flymad_score_movie import OCRThread

class TestTZ:
    def __init__(self, mp4, bag):
        self.bag = bag
        self.t = OCRThread(self._on_processing_finished, 'z', None, OCRThread.MODE_NORMAL, None)
        subprocess.check_call("ffmpeg -i %s -vframes 1 -an -f image2 -y %s" % (mp4,self.t.input_image()),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True)

    def _on_processing_finished(self, err, now, ocrimg, key, tid):
        self.now = now

    def run(self):
        self.t.start()
        self.t.join()

        df = flymad.conv.create_df(self.bag)

        dfnow = df.index[0] / flymad.conv.SECOND_TO_NANOSEC

        print self.now, dfnow

        print (dfnow-self.now) / 60

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mp4', metavar='movie', nargs=1)
    parser.add_argument('bag', metavar='bag', nargs=1)
    args = parser.parse_args()

    t = TestTZ(args.mp4[0], args.bag[0])
    t.run()
