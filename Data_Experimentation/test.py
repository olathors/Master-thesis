import os
import pandas as pd
import numpy as np
#import libitk
import SimpleITK as sitk
import matplotlib.pyplot as plt
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
fi = ants.resample_image(fi, (60,60), 1, 0)
mi = ants.resample_image(mi, (60,60), 1, 0)

ex = ants.registration()