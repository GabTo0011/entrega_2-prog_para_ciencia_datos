#!/usr/bin/env python 
# indica en linux y macOS que este archivo se debe ejecutar con Python
from setuptools import setup, find_packages

# configuramos la carpeta scripts como un paquete de Python
setup(
    name="scripts",
    version="0.1",
    packages=find_packages(),
    description="Paquete local con funciones de preprocesamiento y modelado para ciencia de datos",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn",
    ]
)
