from setuptools import setup, find_packages

setup(
    name="laplace_immo_housing",
    version="1.0.0",
    author="TonNom",
    description="House Price Prediction ML Project",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "catboost",
        "joblib",
        "fastapi",
        "uvicorn"
    ],
    python_requires=">=3.10",
)