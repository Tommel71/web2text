package ch.ethz.dalab.web2text

import ch.ethz.dalab.web2text.utilities.Util
import ch.ethz.dalab.web2text.utilities.Util._
import ch.ethz.dalab.web2text.cleaneval.{CleanEval,Page}
import ch.ethz.dalab.web2text.cdom.{CDOM,DOM}
import org.jsoup.Jsoup
import ch.ethz.dalab.web2text.features.extractor._
import ch.ethz.dalab.web2text.classification.{PerformanceStatistics}
import ch.ethz.dalab.web2text.features.{BlockFeatureExtractor,FeatureExtractor}
import ch.ethz.dalab.web2text.features.PageFeatures
import com.mongodb.casbah.Imports._
import java.io.File;
import ch.ethz.dalab.web2text.output.CsvDatasetWriter
import ch.ethz.dalab.web2text.utilities.Warc
import ch.ethz.dalab.web2text.output.CleanTextOutput
import scala.util.{Try,Success,Failure}

object Main {
  

  // Choose what you want to do in the Main Function.
  def main(args: Array[String]): Unit = {
    // testWarcLoad
    // exportFeaturesTest
    alignCleanEvalData
    //evaluateOtherMethodsCleanEval
    //evaluateOtherMethodsGoogleTrends
  }

  def exportFeaturesTest = {
    val fe = FeatureExtractor(
      DuplicateCountsExtractor
      + LeafBlockExtractor
      + AncestorExtractor(NodeBlockExtractor + TagExtractor(mode="node"), 1)
      + AncestorExtractor(NodeBlockExtractor, 2)
      + RootExtractor(NodeBlockExtractor)
      + TagExtractor(mode="leaf"),
      TreeDistanceExtractor + BlockBreakExtractor + CommonAncestorExtractor(NodeBlockExtractor)
    )
    val data = Util.time{ CleanEval.dataset(fe) }
    CsvDatasetWriter.write(data, "C:/Users/Magnus/Documents/GitHub/web2text/output")
    println("# Block features")
    fe.blockExtractor.labels.foreach(println)
    println("# Edge features")
    fe.edgeExtractor.labels.foreach(println)
  }

  def testCommonAncestorExtractor = {
    val ex = CommonAncestorExtractor(LeafBlockExtractor)
    val cdom = CDOM.fromHTML("<body><h1>Header</h1><p>Paragraph with an <i>Italic</i> section.</p></body>");
    ex(cdom)(cdom.leaves(2),cdom.leaves(1))
  }

  def evaluateOtherMethodsCleanEval = {
    val dir = "other_frameworks/output/"
    val cleaners = Iterable(
      "original"               -> ((id: Int) => s"$dir/original/$id.txt")/*,
      "bte"               -> ((id: Int) => s"$dir/bte/$id-aligned.txt"),
      "article-extractor" -> ((id: Int) => s"$dir/article-extractor/$id-aligned.txt"),
      "default-extractor" -> ((id: Int) => s"$dir/default-extractor/$id-aligned.txt"),
      "largest-content"   -> ((id: Int) => s"$dir/largestcontent-extractor/$id-aligned.txt"),
      "unfluff"           -> ((id: Int) => s"$dir/unfluff/$id-aligned.txt")*/
    )

    for ((label, filenameGen) <- cleaners) {
      val title = s"#### Evaluating ‘${label.capitalize}’ "
      println(s"\n$title${"#"*(82-title.length)}\n")
      Util.time {
        val eval = CleanEval.evaluateCleaner(filenameGen)
        println(s"$eval")
      }
    }
  }

  def evaluateOtherMethodsGoogleTrends = {
    val dir = "other_frameworks/output/"
    val cleaners = Iterable(
      "original"               -> ((id: Int) => s"$dir/original/$id.txt")/*,
      "bte"               -> ((id: Int) => s"$dir/bte/$id-aligned.txt"),
      "article-extractor" -> ((id: Int) => s"$dir/article-extractor/$id-aligned.txt"),
      "default-extractor" -> ((id: Int) => s"$dir/default-extractor/$id-aligned.txt"),
      "largest-content"   -> ((id: Int) => s"$dir/largestcontent-extractor/$id-aligned.txt"),
      "unfluff"           -> ((id: Int) => s"$dir/unfluff/$id-aligned.txt")*/
    )

    for ((label, filenameGen) <- cleaners) {
      val title = s"#### Evaluating ‘${label.capitalize}’ "
      println(s"\n$title${"#"*(82-title.length)}\n")
      Util.time {
        val eval = GoogleTrends.evaluateCleaner(filenameGen)
        println(s"$eval")
      }
    }
  }

  def alignCleanEvalData = {
    val projectPath = "C:\\Users\\Magnus\\Documents\\GitHub\\web2text\\"
    val dir = s"$projectPath/src/main/resources/cleaneval/aligned"
    CleanEval.generateAlignedFiles(dir)
  }
}



