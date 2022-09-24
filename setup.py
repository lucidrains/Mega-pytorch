from setuptools import setup, find_packages

setup(
  name = 'Mega-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.10',
  license='MIT',
  description = 'Mega - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/Mega-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanism',
    'exponential moving average',
    'long range arena'
  ],
  install_requires=[
    'einops>=0.4',
    'scipy',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
