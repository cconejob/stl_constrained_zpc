from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Read the long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='stl_constrained_zpc',
    version='0.1.2',
    packages=find_packages(),
    install_requires=required,  # Use the list from requirements.txt
    entry_points={
        'console_scripts': [
            'stl_constrained_zpc=stl_constrained_zpc.scripts.stl_constrained_zpc:main',
        ],
    },
    author='Carlos Conejo',
    author_email='carlos.conejo@upc.edu',
    description='Zonotopic Predictive Control with Signal Temporal Logic constraints for autonomous vehicles',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cconejob/stl_constrained_zpc.git',
    license='GNU GPLv3',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    include_package_data=True,
)