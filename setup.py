from setuptools import setup, find_packages

setup(
    name="ff-fedtiming-portfolio-optimizer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Portfolio optimization integrating Fama-French factors with Fed policy timing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FF-FedTiming-Portfolio-Optimizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.3.0",
        "yfinance>=0.1.70",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "cvxpy>=1.1.0",
        "beautifulsoup4>=4.10.0",
        "requests>=2.26.0",
        "jupyter>=1.0.0",
        "pytest>=6.2.0",
    ],
    entry_points={
        "console_scripts": [
            "ffopt=src.main:main",
        ],
    },
)
