### Data Science - Photometric Astronomy
---

On this project will have all my codes that was designed as part to obtaing my PhD on Astrophysics. 

I have been working on a study of Brightest Cluster Galaxies (BCGs), which are elliptical galaxies of extreme importance in astrophysics. These objects are located at the centers of galaxy clusters and are characterized by their high mass, extreme luminosity, and a set of properties that differ significantly from those of non-central elliptical galaxies with comparable mass and size.

To investigate these systems, I employed photometric techniques to estimate their structural parameters and to better understand their evolutionary history, particularly in terms of mass growth driven by galactic cannibalism.

As a first step, I developed both 1D and 2D parametric mathematical models to characterize the light distribution of these galaxies. This process involves several stages: data acquisition via `shell` access to the [SDSS](https://www.sdss.org/) database, preprocessing of the raw data using [SExtractor](https://github.com/ICRAR/pyprofit), construction of auxiliary data products such as boolean masks to exclude pixels not associated with the target galaxy, and finally, model fitting.

The 2D modeling itself was carried out using two complementary statistical approaches: a deterministic method implemented with [GALFIT](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html), and a Bayesian framework based on the Bayesian Information Criterion (BIC), using [Profit](https://github.com/ICRAR/ProFit) through its Python wrapper [PyProfit](https://github.com/ICRAR/pyprofit). Whereas for 1D approach was employed the [Photutils](https://photutils.readthedocs.io/en/stable/) that presents functions made especifically for this purpose. 

Both types of modelling (1D and 2D) was invoked from some pipelines (which can be found on the [Model-Evaluation & Build-up](https://github.com/andrelpkaipper/Data-Science---Python-/tree/main/Model-%20Evaluation%20%26%20Build-up)) build on Python language. Nonetheless the programs that was invoked through the pipeline were builded using another languages such as C+ and R.   

That said, the codes won't be specifically used only on my research, one of my goals with this project is to aid and assist another students and researchs a like to deploy or at least have ideias with my codes to theirs on work and studies.


---
~Therefore any advice or assist to understand my codes or any thought about this subject open an issue or send an email to pompeokaipper@gmail.com. I will try to answer as fast as possible.~

