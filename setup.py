import setuptools
# from gastrseg.segm_app import __MAJOR__, __MINOR__, __MICRO__, __AUTHOR__, __VERSION__, __NAME__, __EMAIL__

# with open('README.md', 'r') as fh:
#     long_description = fh.read()

# print(setuptools.find_packages())

setuptools.setup(
    name='MOrgAna',
    version="0.1.0",
    # author=__AUTHOR__,
    # author_email=__EMAIL__,
    description='A machine learning tool to segment organoids.',
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    # url='https://github.com/baigouy/EPySeg',
    # package_data={'': ['*.md']}, # include all .md files
    # license='BSD',
    include_package_data=True,
    packages=setuptools.find_packages(),
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: BSD License',
    #     'Operating System :: OS Independent',
    # ],

    install_requires=[
        # "tensorflow>=2.4",  # to allow for mac OS X conda support #shall I put 2.3 now
        # "Markdown",
        "matplotlib",
        "numpy",
        # "PyQt5",
        "scikit-image",
        "scipy",
        "tqdm",
    ],
    # extras_require = {
    #     'all':  ["tensorflow-gpu>=2.0.0"]
    # },
    python_requires='>=3.6' # tensorflow is now supported by python 3.8
)

