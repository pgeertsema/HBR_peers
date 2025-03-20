# HBR_peers
This repo contains code to generate the results in the paper "*AI brings a fresh approach to relative valuation*" by Geertsema, Lu and Stouthuysen (2025), forthcoming in the Harvard Business Review.

The code consists of two files:

1. [initial_data_processing.do](initial_data_processing.do) is a Stata file that handles the basic data preparation and cleaning needed.
2. [create_ml_results.py](create_ml_results.py) is a Python file that generates the images and results presented in the paper.

Both files contain copious comments that serves as documentation of our methodology.

For data we rely on the open source data from Chen and Zimmerman (2021)
[https://www.openassetpricing.com/](https://www.openassetpricing.com/)
see "Open Source Cross-Sectional Asset Pricing" in Critical Finance Review 
[https://www.nowpublishers.com/article/Details/CFR-0112](https://www.nowpublishers.com/article/Details/CFR-0112)

We also use CRSP data ([https://www.crsp.org/](https://www.crsp.org/)) for market capitalisation and tickers; however, this is not essential. (Market cap is used to select large firms as an initial universe, while tickers are used to create human-readable labels for the firms)

To calculate peer-weights, we rely on a copy of the AXIL code available at [https://github.com/pgeertsema/AXIL_paper](https://github.com/pgeertsema/AXIL_paper)
This is explained in more detail in the paper "*Instance-based Explanations for Gradient Boosting Machine Predictions with AXIL Weights*" by Geertsema & Lu (2023): [https://arxiv.org/abs/2301.01864](https://arxiv.org/abs/2301.01864)

The concept of peer-weights in the context of valuation was originally developed in the paper "*Relative valuation with Machine Learning*" by Geertsema and Lu (2023) published in the Journal of Accounting Research, see [https://onlinelibrary.wiley.com/doi/full/10.1111/1475-679X.12464](https://onlinelibrary.wiley.com/doi/full/10.1111/1475-679X.12464).
Code for that paper is available at [https://github.com/helenhelu/RelativeValuationWithMachineLearning](https://github.com/helenhelu/RelativeValuationWithMachineLearning)

For any questions, feel free to reach out to me at paul.geertsema /at/ vlerick.com
