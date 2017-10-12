import numpy as np

head = [
    ('run', np.uint16),
    ('event', np.uint16),
    ('reuse', np.uint16),

    ('azimuth', np.float16),
    ('zenith', np.float16),

    ('number_photons', np.float16),
    ('cog_cx_pap', np.float16),
    ('cog_cy_pap', np.float16),

    ('energy', np.float16),
    ('theta', np.float16),
    ('phi', np.float16),
    ('impact_x', np.float16),
    ('impact_y', np.float16),
    ('particle', np.uint16),
    ('hight_of_first_interaction', np.float16),
]

sorted_keys = [
    ('number_photons', np.float16),
    ('cog_cx_pap', np.float16),
    ('cog_cy_pap', np.float16),   
]
