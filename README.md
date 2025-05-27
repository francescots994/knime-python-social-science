# Statistics and Social Science Extension for KNIME

This repository contains the code for the Statistics and Social Science Extension for the KNIME Analytics Platform. This extension provides nodes for advanced statistical analysis, time series modeling, and data visualizations, enabling users to perform in-depth explorations of structured data.

The extension is curated and maintained by Francesco Tuscolano (KNIME), Prof. Daniele Tonini, and Pietro Maran (Bocconi University, Milan).

The project's goal is to integrate advanced statistical methodologies within KNIME by leveraging bundled Python packages and transforming them into native KNIME nodes.

## Current Nodes

* Auto-SARIMA Learner: Automatically finds the optimal parameters for and trains a Seasonal AutoRegressive Integrated Moving Average (SARIMA) model on a given time series, returning also the in-sample predictions, residuals and model statistics. This model is the SARIMAX class from the statsmodels library. 
* Auto-SARIMA Predictor: Generates future forecasts using a pre-trained SARIMA model, output of the node Auto-SARIMA Learner.

## Package Organization

* `icons`: Folder containing the all images in the extension.
* `config.yml`: Example `config.yml` to point to the directory containing the source code of the extension. This directory must also contain `knime.yml` file.
* `knime.yml`: YAML file with information on the extension.
* `src/social_science_ext.py`: The python file with the knime details of the extension.
* `src/nodes/`: Contains the source codes of the two nodes in the extension, the Auto-SARIMA Learner node, to find the optimal model, and the Predictor node, for forecast.
* `src/util/`: Contains the utils.py file, with several utility functions are reused from Harvard's spatial data lab repository for Geospatial Analytics Extension.