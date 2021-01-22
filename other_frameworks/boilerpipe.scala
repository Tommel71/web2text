import java.io.FileReader
import de.l3s.boilerpipe.extractors._

import ch.ethz.dalab.web2text.cleaneval._
import ch.ethz.dalab.web2text.alignment._
import ch.ethz.dalab.web2text.utilities._

object Boilerpipe {

  def main(args: Array[String]): Unit = {
    cleanPages
    alignPages
  }

  /** Step 1: cleans the pages and saves them in the `output` dir */
  def cleanPages = {

    for (page <- CleanEval.iterator) {
      val text = page.source
      val cleaned = LargestContentExtractor.INSTANCE.getText(text);
      Util.save(s"output/largestcontent-extractor/${page.id}-clean.txt", cleaned)
      val cleaned2 = ArticleExtractor.INSTANCE.getText(text);
      Util.save(s"output/article-extractor/${page.id}-clean.txt", cleaned2)
      val cleaned3 = DefaultExtractor.INSTANCE.getText(text);
      Util.save(s"output/default-extractor/${page.id}-clean.txt", cleaned3)
      println(s"Done with ${page.id}")
    }

  }

  /** Step 2: Align the files and save them in the `output` dir as well */
  def alignPages = {
    for(folder <- List("largestcontent-extractor","default-extractor","article-extractor")) {
    val ooms = new scala.collection.mutable.ArrayBuffer[Int]()
    for (page <- CleanEval.iterator) {
      try {
        val orig = page.source
        val clean = Util.loadFile(s"output/$folder/${page.id}-clean.txt").trim
        
        if (!Util.fileExists(s"output/$folder/${page.id}-aligned.txt")) {
          try {
            val alignment = Alignment.alignment(orig, clean)
            Util.save(s"output/$folder/${page.id}-aligned.txt", alignment)
            println(s"Done with ${page.id}")
          } catch {
            case e: OutOfMemoryError => {
              println(s"Error: OOM: ${page.id}")
              ooms+=page.id.toInt
              }
            case e: NegativeArraySizeException => {
              println(s"Error: NegArray: ${page.id}")
              ooms+=page.id.toInt
              }
          }
        }
        else {
          println(s"${page.id}-aligned already there.")
        }
       
      } catch {
          case e: NullPointerException => {
              println(s"Error: NullPointer: ${page.id}");
              ooms+=page.id.toInt
              }
      } 
    }
    println("Hier suchen")
    println(ooms)
    }
  }

}
