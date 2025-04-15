from setuptools import setup, find_packages

setup(
    name="adiance-frvt",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'opencv-python>=4.5.0',
        'onnxruntime>=1.8.0',
        'flask>=2.0.0',
        'flask-limiter>=2.0.0',
        'werkzeug>=2.0.0',
    ],
    python_requires='>=3.7',
    author="Your Name",
    author_email="your.email@example.com",
    description="A face recognition system using RetinaFace and AdaFace",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
