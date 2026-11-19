from setuptools import setup, find_packages

setup(
    name="lunar_registration",
    version="0.1.0",
    # Backend packages (api, src) live under server/
    package_dir={"": "server"},
    packages=find_packages(where="server"),
    install_requires=[
        "opencv-python",
        "opencv-contrib-python",
        "torch",
        "kornia",
        "scikit-image",
        "numpy",
        "pandas",
        "streamlit",
        "spiceypy",
        "pds4_tools",
        "rasterio"
    ],
    entry_points={
        'console_scripts': [
            'lunar-reg=src.pipeline:main',
        ],
    },
)
