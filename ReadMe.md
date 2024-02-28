# RVultra

Do you have:
- A bunch of RVs, potentially from a bunch of different instruments?

Do you want to:
- Compute a no-frills RV model?
- Calculate a robust mass upper limit?
- Use the transit duration to help constrain eccentricity (and therefore K) without relying on a fully combined photometry+RV model?
- Use nested sampling to marginalise over nuisance parameters?
- Compute the Bayesian evidence for the planet model compared to a no-planet model?

Then *RVUltra* is for you. It uses Ultranest but hides the silly syntax in an easy-to-use pythonic UI for use in scripts and notebooks.

To see an example try [here](https://github.com/hposborn/RVultra/blob/main/RVultra%20Example.ipynb) for which the output can be found here: 

![G 75-21 RVs with planetary models](https://github.com/hposborn/RVultra/blob/main/G%2075-21fit_rvmodel_plot.png?raw=true)

The code builds on work from [radvel](https://radvel.readthedocs.io/en/latest/), [Ultranest](https://johannesbuchner.github.io/UltraNest/index.html) as well taking style hints from my other projects [MonoTools](https://github.com/hposborn/MonoTools) and [chexoplanet](https://github.com/hposborn/chexoplanet).