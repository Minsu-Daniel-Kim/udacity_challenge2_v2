import scipy.misc
import time
import cv2
import os
from subprocess import call
import numpy as np
import json
import logging
import argparse
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Let's do it Yeon hoo")

parser = argparse.ArgumentParser()
parser.add_argument("--path")
args = parser.parse_args()

i = 0



processed_pickles = [int(item.split(".")[0]) for item in os.listdir(args.path) if item.endswith(".png")]
processed_pickles.sort()
# processed_pickles = np.array(processed_pickles)
processed_pickles = [str(item) + '.png' for item in processed_pickles]
print(processed_pickles)

frame_to_be_removed = []
removing_in_progress = False

while(cv2.waitKey(10) != ord('q')):
    time.sleep(0.01)
    full_image = scipy.misc.imread(args.path + '/' + processed_pickles[i], mode="RGB")

    if cv2.waitKey(10) is ord('a'):
        logger.info("let's remove from here!")
        removing_in_progress = not removing_in_progress
    if removing_in_progress:

        logger.info('removing is progress...')
        print('removing is progress...')

        frame_to_be_removed.append(processed_pickles[i])
        full_image = full_image[:, :, 1] - 100
        cv2.imshow("Video View", cv2.cvtColor(full_image, cv2.COLOR_BAYER_BG2GRAY))
    else:
        cv2.imshow("Video View", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    call("clear")
    i += 1

print(frame_to_be_removed)

bundle = {
    
    'last_index': i,
    'frame_to_be_removed': frame_to_be_removed

}
with open('~/Desktop/frame_to_be_removed.txt', 'w') as outfile:
    json.dump(bundle, outfile)




cv2.destroyAllWindows()