import numpy as np

head = [
    ('run', np.uint16),
    ('event', np.uint16),
    ('reuse', np.uint16),

    ('telescope_az', np.float32),
    ('telescope_zd', np.float32),

    ('number_photons', np.float16),
    ('cog_cx_pap', np.float16),
    ('cog_cy_pap', np.float16),

    ('energy', np.float16),
    ('source_az', np.float32),
    ('source_zd', np.float32),
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
