try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'My Neural Network Project',
    'author': 'Nacho Villaluenga',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'nachovillaluenga@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['NN'],
    'scripts': [],
    'name': 'ProjectName'
}

setup(**config)
