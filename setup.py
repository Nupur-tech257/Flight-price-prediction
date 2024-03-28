from setuptools import find_packages,setup
from typing import List

HYPHEN_E= "-e ."
def get_file(filepath:str)->List[str]:
    """this function returns the list of all the requirements mentioned in the file"""
    requirements=[]
    with open(filepath) as file_obj:
        requirements=file_obj.readlines()
        req=[i.replace("\n"," ") for i in requirements]
        if HYPHEN_E in req:
            req.remove(HYPHEN_E)
    return req

setup(
name='ml_project',
version='0.0.1',
author='Nupur',
author_email='sk08251977@gmail.com',
packages=find_packages(),
install_requires=get_file('requirements.txt')
)
