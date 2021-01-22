import java.io.FileReader
import de.l3s.boilerpipe.extractors._

import ch.ethz.dalab.web2text.cleaneval._
import ch.ethz.dalab.web2text.alignment._
import ch.ethz.dalab.web2text.utilities._

object Unfluff {

  def main(args: Array[String]): Unit = {
    alignPages
  }

  /** Step 1: Generate the prediction files, which the `unfluff/unfluff.js` command line tool.
    * Make sure to set the right in / output directories */

  /** Step 2: Align the files and save them in the `output` dir as well */
  def alignPages = {
    val folder = "unfluff"
    val ooms = new scala.collection.mutable.ArrayBuffer[Int]()
    for (page <- CleanEval.iterator) {
      try {
        val orig = page.source
        val clean = Util.loadFile(s"output/$folder/${page.id}.html").trim
        
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
    println(ooms)
  }

}
