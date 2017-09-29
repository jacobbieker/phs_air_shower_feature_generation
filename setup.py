from distutils.core import setup

setup(
    name='photon_stream_analysis',
    version='0.0.0',
    description='generate air-shower features from the photon-stream',
    url='https://github.com/fact-project/',
    author='Sebastian Achim Mueller',
    author_email='sebmuell@phys.ethz.ch',
    license='GPLv3',
    packages=[
        'photon_stream_analysis',
    ],
    package_data={
        'photon_stream_analysis': [
            'tests/resources/*',
        ]
    },
    install_requires=[
        'pyfact',
        'pandas',
    ],
    zip_safe=False,
)
