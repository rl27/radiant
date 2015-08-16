#!/usr/bin/env python

from setuptools import setup

# class to clean the build leftovers
class CleanRadiant(clean):
    """
        Remove build directories and leftover files from compilation.
    """

    def run(self):
        """
            Run the clean operation.
        """
        Clean.run(self)
        if os.path.exists("radiant.egg-info"):
            shutil.rmtree("radiant.egg-info")
        if os.path.exists("build"):
            shutil.rmtree("build")
        if os.path.exists("dist"):
            shutil.rmtree("dist")


if __name__ == "__main__":

    # version
    with open("VERSION.txt") as f:
        version = f.read().strip()

    # long description
    with open("DESCRIPTION.txt") as f:
        long_description = f.read().strip()

    # requirements
    with open("requirements.txt") as f:
        install_requires = [l.strip() for l in f.readlines()]

    # setup
    setup(
        name='radiant',
        version=version,
        description='An advanced set of tools for image editing.',
        long_description=long_description,
        url='https://github.com/fzliu/radiant',
        author='Frank Liu',
        author_email='frank.zijie@gmail.com',
        license='Apache',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        packages = ['radiant',
                    'radiant.art',
                    'radiant.data',
                    'radiant.filter',
                    'radiant.misc',
                    'radiant.resize'
        ],
        install_requires = install_requires,
        zip_safe=False,
)