package ch.ethz.dalab.web2text.cleaneval

import ch.ethz.dalab.web2text.utilities.Util
import ch.ethz.dalab.web2text.alignment.Alignment
import ch.ethz.dalab.web2text.features.{FeatureExtractor,PageFeatures}
import ch.ethz.dalab.web2text.cdom.CDOM
import ch.ethz.dalab.web2text.classification.PerformanceStatistics
import ch.ethz.dalab.web2text.utilities.Util.codec
import scala.io.{Source,Codec}
import org.jsoup.Jsoup

/** GoogleTrends related functionality
  *
  * @author Thijs Vogels <t.vogels@me.com>
  */
object GoogleTrends {

  /** Directory within `src/main/resources` in which to find the googletrends data */
  val directory = "/googletrends"

  /** Get path to the resource of the HTML source for document #n */
  def origPath(n: Int) = s"$directory/orig/$n.html"

  /** Get path to the resource of the cleaned document (gold standard) for document #n */
  def cleanPath(n: Int) = s"$directory/clean/$n.txt"

  /** Get path to the resource of the cleaned aligned document for document #n.
    * @see [[ch.ethz.dalab.web2text.alignment.Alignment]]. */
  def alignedPath(n: Int) = s"$directory/aligned/$n.txt"


  /** Vector of indices of googletrends items that are complete with source, cleaned and aligned versions. */
  val indices: Vector[Int] = Vector(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180)

  /** Get contents of the cleaned file with ID `i`
    *
    * Potential first lines containing URL: http://... are removed.
    */
  def loadCleanFile(i: Int): String =
    loadCleanFile(cleanPath(i), isResource=true, normalize=true)

  /** Get contents of the HTML source with ID `i` */
  def loadOrigFile(id: Int): String = {
    val path = origPath(id)
    val stream = getClass.getResourceAsStream(path)
    val source = Source.fromInputStream(stream)
    val lines = source.getLines.drop(1).toVector
    lines.slice(0,lines.length-1) mkString "\n"
  }

  /** Get contents of the aligned clean version with ID `i` */
  def loadAlignedFile(i: Int): String = Util.loadFile(alignedPath(i), isResource=true)

  /** Get contents of a cleaned file, stored anywhere
    *
    * Potential first lines containing URL: http://... are removed.
    */
  def loadCleanFile(path: String, normalize: Boolean = false, isResource: Boolean = false): String = {
    val f = Util.loadFile(path, isResource=isResource)

    val contents = if (f.startsWith("URL:")) f.lines.drop(1).mkString("\n")
                   else f

    if (normalize)
      normalizeCleanFile(contents)
    else
      contents
  }

  def loadFirstLine(id: Int) = {
    try {
    val path = GoogleTrends.origPath(id)
    val stream = getClass.getResourceAsStream(path)
    val source = Source.fromInputStream(stream)
    source.getLines.next()
    } catch {
      case e: NullPointerException => ""
    }
  }

  /** Normalize a cleaned file by removing <H> <P> <L> and list markings from the file
    * and trimming it.
    */
  def normalizeCleanFile(txt: String): String =
    txt
       .replaceAll("""(?i)(?m)^\s*<l>\s*(»|\*|\d{1,2}\.\s)\s*|<(l|h|p)>\s*|^\s*(_{10,}|-{10,})\s*$|^\s*""","")
       .trim


  /** Create an iterator that produces the GoogleTrends pages with all available content one by one. */
  def iterator: Iterator[Page] =
    indices.toIterator.map {i => Page(i) }

  /** Generate a dataset for training / testing a classifier
    * @param take How many documents to use (-1 = all) */
  def dataset(extractor: FeatureExtractor, take: Int = -1): Vector[(PageFeatures,Vector[Int])] = {

      val it = if (take == -1) iterator else iterator.take(take)

      val result = it map { p =>
        val cdom = CDOM(p.source)
        val features = extractor(cdom)
        val labels = Alignment.extractLabels(cdom, p.aligned)
        (features,labels)
      }

      result.toVector
  }

  /** Evaluate a cleaner by its aligned files.
    * If a cleaned file for a certain index is missing, it is skipped without penalty.
    * @param alignedLocation function turning an index to the file location. */
  def evaluateCleaner(alignedLocation: Int=>String): PerformanceStatistics = {
    val pairs = iterator flatMap { p =>
      val path = alignedLocation(p.id)
      if (!Util.fileExists(path)) {
        Vector()
      } else {
        import Alignment.extractLabels
        val cdom        = CDOM(p.source)
        val goldLabels  = extractLabels(cdom, p.aligned)
        val otherLabels = extractLabels(cdom, Util.loadFile(path))
        (otherLabels zip goldLabels)
      }
    }
    PerformanceStatistics.fromPairs(pairs.toVector)
  }



}
