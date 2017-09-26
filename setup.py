from distutils.core import setup

setup(
    name='phs_air_shower_feature_generation',
    version='0.0.0',
    description='generate air-shower features from the photon-stream',
    url='https://github.com/fact-project/',
    author='Sebastian Achim Mueller',
    author_email='sebmuell@phys.ethz.ch',
    license='GPLv3',
    packages=[
        'phs_air_shower_feature_generation',
    ],
    package_data={
        'phs_air_shower_feature_generation': [
            'tests/resources/*',
        ]
    },
    install_requires=[
        'pyfact',
        'pandas',
    ],
    zip_safe=False,
)
