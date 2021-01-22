# Web2Text

Source code for [Web2Text: Deep Structured Boilerplate Removal](https://arxiv.org/abs/1801.02607), full paper at ECIR '18 

## Introduction

This repository contains 

* Scala code to parse an (X)HTML document into a DOM tree, convert it to a CDOM tree, interpret tree leaves as a sequence of text blocks and extract features for each of these blocks. 

* Python code to train and evaluate unary and pairwise CNNs on top of these features. Inference on the hidden Markov model based on the CNN output potentials can be executed using the provided implementation of the Viterbi algorithm.

* The [CleanEval](https://cleaneval.sigwac.org.uk) dataset under `src/main/resources/cleaneval/`:
    - `orig`: raw pages
    - `clean`: reference clean pages
    - `aligned`: clean content aligned with the corresponding raw page on a per-character basis using the alignment algorithm described in our paper

* Output from various other webpage cleaners on CleanEval under `other_frameworks/output`:
    - [Body Text Extractor](https://www.researchgate.net/publication/2376126_Fact_or_fiction_Content_classification_for_digital_libraries) (Finn et al., 2001)
    - [Boilerpipe](https://github.com/janih/boilerpipe) (Kohlschütter et al., 2010): default-extractor, article-extractor, largestcontent-extractor
    - [Unfluff](https://github.com/ageitgey/node-unfluff) (Geitgey, 2014)
    - [Victor](https://pdfs.semanticscholar.org/5462/d15610592394a5cd305d44003cc89630f990.pdf) (Spousta et al., 2008)



## Installation

1. Install [Scala and SBT](http://www.scala-sbt.org/download.html). The code was tested with SBT 0.31. You can also use Docker image `hseeberger/scala-sbt:8u222_1.3.3_2.13.1`.

2. Install Python 3 with Tensorflow and NumPy.


## Usage

### Recipe: extracting text from a web page

1. Run `ch.ethz.dalab.web2text.ExtractPageFeatures` through sbt. The arguments are:
    * input html file
    * the desired output base filename (script produces `{filename_base}_edge_feature.csv` and `{filename_base}_block_features.csv`)
2. Use the python script `src/main/python.py` with the 'classify' option. The arguments are:
    * `python3 main.py classify {filename_base} {labels_out_filename}`
2. Use `ch.ethz.dalab.web2text.ApplyLabelsToPage` through sbt to produce clean text. Arguments:
    * input html file
    * `{labels_out_filename}` from step 2
    * output destination text file path


### HTML to CDOM

In Scala:

```scala
import ch.ethz.dalab.web2text.cdom.CDOM
val cdom = CDOM.fromHTML("""
    <body>
        <h1>Header</h1>
        <p>Paragraph with an <i>Italic</i> section.</p>
    </body>
    """)
println(cdom)
```

### Feature extraction

Example:
```scala
import ch.ethz.dalab.web2text.features.{FeatureExtractor, PageFeatures}
import ch.ethz.dalab.web2text.features.extractor._

val unaryExtractor = 
    DuplicateCountsExtractor
    + LeafBlockExtractor
    + AncestorExtractor(NodeBlockExtractor + TagExtractor(mode="node"), 1)
    + AncestorExtractor(NodeBlockExtractor, 2)
    + RootExtractor(NodeBlockExtractor)
    + TagExtractor(mode="leaf")

val pairwiseExtractor = 
    TreeDistanceExtractor + 
    BlockBreakExtractor + 
    CommonAncestorExtractor(NodeBlockExtractor)

val extractor = FeatureExtractor(unaryExtractor, pairwiseExtractor)

val features: PageFeatures = extractor(cdom)

println(features)
```

### Aligning cleaned text with original source

```scala
import ch.ethz.dalab.web2text.alignment.Alignment
val reference = "keep this"
val source = "You should keep this text"
val alignment: String = Alignment.alignment(source, reference) 
println(alignment) // □□□□□□□□□□□keep this□□□□□
```
### Extracting features for CleanEval

```scala
import ch.ethz.dalab.web2text.utilities.Util
import ch.ethz.dalab.web2text.cleaneval.CleanEval
import ch.ethz.dalab.web2text.output.CsvDatasetWriter

val data = Util.time{ CleanEval.dataset(fe) }

// Write block_features.csv and edge_features.csv
// Format of a row: page id, groundtruth label (1/0), features ...
CsvDatasetWriter.write(data, "./src/main/python/data")

// Print the names of the exported features in order
println("# Block features")
fe.blockExtractor.labels.foreach(println)
println("# Edge features")
fe.edgeExtractor.labels.foreach(println)
```

### Training the CNNs

Code related to the CNNs lives in the `src/main/python` directory. 

To train the CNNs:

1. Set the `CHECKPOINT_DIR` variable in `main.py`.
2. Make sure the files `block_features.csv` and `edge_features.csv` are in the `src/main/python/data` directory. Use the example from the previous section for this.
3. Convert the CSV files to `.npy` with `data/convert_scala_csv.py`.
3. Train the unary CNN with `python3 main.py train_unary`.
4. Train the pairwise CNN with `python3 main.py train_edge`.

### Evaluating the CNN

To evaluate the CNN:

1. Set the `CHECKPOINT_DIR` variable in `main.py` to point to a directory with trained weights. We provide trained weights based on the cleaneval split and a custom web2text split (with more training data.)
2. Run `python3 main.py test_structured` to test performance on the CleanEval test set.


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
Afterwards, the command "run" opens a list of runnable programs.

### Boilerpipe
https://code.google.com/archive/p/boilerpipe/

Run "Boilerpipe" in SBT.


### BTE
https://github.com/girish/utils/blob/master/text_extraction/bte.py

1. Generate the prediction files with "bte.py"
2. Run "BTE" in SBT.


### Unfluff
https://github.com/ageitgey/node-unfluff

1. Install Node.
2. Run the following commands in a Shell in the "other_frameworks" folder
3. `npm install -g unfluff`
4. `node unfluff/unfluff.js'
5. Run "Unfluff" in SBT.

### Performance-Metrics

For the PerformanceMetrics, run "Main.scala" from the original project with the function "evaluateOtherMethods" inside of the main-Function. The results appear in the console.
