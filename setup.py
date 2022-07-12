import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# print(setuptools.find_packages())

setuptools.setup(
    name='morgana',
    version="0.1.1",
    author="Nicola Gritti",
    author_email="gritti@embl.es",
    description="A machine learning tool to segment organoids.",
    long_description= long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/LabTrivedi/MOrgAna',
    # package_data={'': ['*.md']}, # include all .md files
    # license='BSD',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # package_dir={"": "src"},
    packages=setuptools.find_packages(),#where="src"),
    install_requires=[
        # "tensorflow>=2.4",  # to allow for mac OS X conda support #shall I put 2.3 now
        # "Markdown",
        "matplotlib",
        "numpy>=1.20",
        # "PyQt5",
        "scikit-image>=0.18",
        "pandas>=1.2",
        "joblib>=1.0",
        "scikit-learn>=0.24",
        "scipy>=1.6",
        "tqdm>=4.60",
    ],
    # extras_require = {
    #     'all':  ["tensorflow-gpu>=2.0.0"]
    # },
    python_requires='>=3.6', # tensorflow is now supported by python 3.8
    zip_safe=False 
)

