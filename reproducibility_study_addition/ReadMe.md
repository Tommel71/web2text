## Code Description

### data.py

This file was only slightly changed from from the original by adding support for the GoogleTrends-2017 dataset. Therefore, we added a variable containing the GoogleTrends Numpy Data and an array containing all data.

### bootstrap.py

In this file we added the support to perform Repeated Random Sampling (Bootstrap) in order to generate multiple results with different train/val and test data. Therefore, we reused the methods implemented by the Paper authors to train and test the model and added a function which iterate x-times and generates for every iteration a new random split, trains the models and tests the performance. 

In order to run the script use it that way:

python bootstrap.py num_repetitions dataset

The results are stores in a folder called "results"

### statistical testing.py

data has to be saved like thisfor the file to work:

googletrends structured approach bootstrap results: "results/googletrends/Structured_Results.csv"
googletrends unary approach bootstrap results: "results/googletrends/Unary_Results.csv"
Cleaneval structured approach bootstrap results on original split: "results/cleaneval_original_split/Structured_Results.csv"
Cleaneval unary approach bootstrap results on original split: "results/cleaneval_original_split/Unary_Results.csv"
Cleaneval structured approach bootstrap results on Web2Text split: "results/cleaneval_web2text_split/Structured_Results.csv"
Cleaneval unary approach bootstrap results on Web2Text split: "results/cleaneval_web2text_split/Unary_Results.csv"

This script takes outputs from the bootstrapping in the results folder and does the statistical tests and creates the plots we used in the paper.


### googletrends2017_to_cleaneval.py

This script is used to convert googletrends2017 target files to the cleaneval style of target files.

path_in determines where the googletrends2017 prepared files are located
path_out determines where the extracted data is saved.

### use_alignment.sc

Place this file in the web2text\src\main\resources folder of the Web2Text repository along with the folders "cleaneval_style" and "raw_html" for the output of googletrends2017_to_cleaneval.py and the raw googletrends2017 data.

Execute this code in the Scala REPL (if i just ran the file in IntelliJ the placeholders were questionmarks which makes the results unusable). 

The result is the aligned file which can be used in the Web2Text algorithm.