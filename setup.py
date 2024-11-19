from setuptools import setup, find_packages

setup(name='model_augmentation',
      version='1.0.0',
      description='Model augmentation framework for system identification',
      author='Bendegúz Györök',
      author_email='gyorokbende@sztaki.hu',
      packages=find_packages(),
      install_requires=[
            "numpy==1.24.4",
            "torch",
            "matplotlib",
            "tqdm",
            "deepSI @ git+https://github.com/GerbenBeintema/deepSI@master"
        ]
      )
