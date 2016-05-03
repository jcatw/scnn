__author__ = 'jatwood'


# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='scnn',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.3.1',

    description='Search-Convolutional Neural Networks',
    long_description="""
SCNN
====

An implementation of search-convolutional neural networks [1], a new model for graph-structured data.

Installation
------------
Using pip:

    pip install scnn

Usage
-----

	import numpy as np
    from scnn import SCNN, data
    from sklearn.metrics import f1_score

	# Parse the cora dataset and return an adjacency matrix, a design matrix, and a 1-hot label matrix
    A, X, Y = data.parse_cora()

	# Construct array indices for the training, validation, and test sets
    n_nodes = A.shape[0]
    indices = np.arange(n_nodes)
    train_indices = indices[:n_nodes // 3]
    valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
    test_indices  = indices[(2* n_nodes) // 3:]

	# Instantiate an SCNN and fit it to cora
    scnn = SCNN()
    scnn.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices)

	# Predict labels for the test set
    preds = scnn.predict(X, test_indices)
    actuals = np.argmax(Y[test_indices,:], axis=1)

	# Display performance
    print 'F score: %.4f' % (f1_score(actuals, preds))

References
----------

[1] http://arxiv.org/abs/1511.02136
    """,

    # The project's main homepage.
    url='https://github.com/jcatw/scnn',

    # Author details
    author='James Atwood',
    author_email='james.c.atwood@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords='search convolutional neural networks machine learning',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy',
                      'theano',
                      'lasagne>=0.1dev',
                      'matplotlib',
                      ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={},

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={'scnn': ['README.md',
                           'scnn/data/cora',
                           'scnn/data/Pubmed-Diabetes',
                           'scnn/data/blogcatalog',
                           'scnn/data/nci']},

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={},
)
