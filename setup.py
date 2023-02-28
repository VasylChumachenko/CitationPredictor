from setuptools import setup

setup(
    name='citation_predictor',
    version='0.0.1',    
    description='POC',
    url='https://github.com/shuds13/pyexample',
    author='Vasyl Chumachenko',
    author_email='chumachenko_va@ukr.net',    
    license='BSD 2-clause',
    packages=['citation_predictor'],
    install_requires=['pandas',
                      'numpy',
                      'lightgbm',
                      'sklearn',
                      'shap',
                      'matplotlib',
                      'nltk'],

    classifiers=[
        'Development Status :: 1 - alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)   