import shutil
import os
import argparse
import numpy as np

class remover:

    def __init__(self, path):
        self.path = path
        self.processed_pickles = np.array([int(item.split(".")[0]) for item in os.listdir(path) if item.endswith(".png")])
        self.processed_pickles.sort()

    def d(self, start, end):
        to_be_removed = self.processed_pickles[(self.processed_pickles >= start) & (self.processed_pickles <= end)]
        print("remove :",  to_be_removed)
        for item in to_be_removed:
            os.remove(self.path + '/' + str(item) + '.png')
