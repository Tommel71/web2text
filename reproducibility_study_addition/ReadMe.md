## Code Description

We used Python 3.7.9

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

# Get bootstrap evaluation of GT17 and CleanEval

1. Use `googletrends2017_to_cleaneval.py` on the  GT17 data, to get it into the same form as the cleaneval data set.
2. Use the alignment algorithm of the Web2Text repository by using the `use_alignment.sc` scala script on the cleaneval style groundtruth of GT17. Adapt the paths to your groundtruth (no boilerplate) dataset and the dataset with boilerplate.
3. Running `ch.ethz.dalab.web2text.Main` using the GoogleTrends or CleanEval class respectively, extract the features.
4. Use `bootstrap.py` on the features to get the metrics of Web2Text on the bootstrap samples.
5. Use `statistical testing.py` on the bootstrap metric results.


# Performance of other networks:

* When using other frameworks on CleanEval, the .scala-files in the source folder do not need to be changed.
* When using other frameworks on GT17, the file "CleanEval.scala" in "src\main\scala\ch\ethz\dalab\web2text\cleaneval" needs to be changed according to the comments. 
This results in the algorithms only running on the GT17-indices instead of the CleanEval indices.
The GT17 files should be loaded into "src\main\resources\cleaneval" instead of the CleanEval files.


Before trying to run the different .scala files in the "other_frameworks"-directory, the main scala project needs to be published locally. This can be done with sbt entering the following lines into a Shell opened in the main folder and confirming with Enter after each line.
```scala
sbt
compile
publishLocal
```

Afterwards, the different frameworks can be applied to the dataset. For this, you need to open a Shell inside of the other_frameworks folder. Scala-Files need to be run by entering SBT first:
```scala
sbt 
```
Afterwards, the command `run` opens a list of runnable programs.

### Boilerpipe
https://code.google.com/archive/p/boilerpipe/

Run `Boilerpipe` in SBT.


### BTE
https://github.com/girish/utils/blob/master/text_extraction/bte.py

1. Generate the prediction files with `bte.py`
2. Run `BTE` in SBT.


### Unfluff
https://github.com/ageitgey/node-unfluff

1. Install Node.
2. Run the following commands in a Shell in the `other_frameworks` folder
3. `npm install -g unfluff`
4. `node unfluff/unfluff.js`
5. Run `Unfluff` in SBT.

### Performance-Metrics

For the PerformanceMetrics, run `Main.scala` from the original project with the function `evaluateOtherMethods` inside of the main-Function. The results appear in the console.

