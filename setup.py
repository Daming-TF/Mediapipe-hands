from setuptools import setup, find_packages

setup(
    name="lib",
    packages=find_packages(),
    version="0.1",
    license="MIT",
    description="Hand Landmark and Gesture Recognition (HLGR)",
    author="Huya Inc",
    author_email="jinchengbin@huya.com",
    url="https://aigit.huya.com/jinchengbin/mediapipe-hands",
    keywords=[
        "hand landmarks",
        "hands gesture recognition",
        "hand detection",
        "pose landmarks",
        "mediapipe",
    ],
    python_requires=">=3.6",
    install_requires=["opencv-python", "onnxruntime-gpu", "pre-commit", "black"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
