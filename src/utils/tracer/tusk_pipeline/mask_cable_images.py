import numpy as np
import os

input_folder = '../data/real_data/real_data_for_tracer/test'
output_folder = '../data/real_data/real_data_for_tracer/test_masked'

for i, data in enumerate(np.sort(os.listdir(input_folder))):
    test_data = np.load(os.path.join(input_folder, data), allow_pickle=True).item()
    img = test_data['img']
    import pdb; pdb.set_trace()




