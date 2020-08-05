
import codecs
from setuptools import setup, find_packages

with codecs.open('README.md', 'r', 'utf8') as reader:
    long_description = reader.read()


with codecs.open('requirements.txt', 'r', 'utf8') as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))


setup(
    name='bert_multitask_learning',
    version='0.3.2',
    packages=find_packages(),
    url='https://github.com/JayYip/bert-multitask-learning',
    license='MIT',
    author='Jay Yip',
    author_email='junpang.yip@gmail.com',
    description='BERT for Multi-task Learning',
    long_description_content_type='text/markdown',
    long_description=long_description,
    python_requires='>=3.5.0',
    install_requires=install_requires,
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
    ),
)
