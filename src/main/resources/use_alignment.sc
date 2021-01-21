import ch.ethz.dalab.web2text.alignment.Alignment
import ch.ethz.dalab.web2text.utilities.Util
import java.io.File

def getListOfFiles(dir: String):List[File] = {
  val d = new File(dir)
  if (d.exists && d.isDirectory) {
    d.listFiles.filter(_.isFile).toList
  } else {
    List[File]()
  }
}

val filelist = getListOfFiles("C:\\Users\\Tom\\Documents\\GitHub\\web2text\\src\\main\\resources\\cleaneval_style")
for(x <- filelist){
  val name_end = x.getName()
  print("\n")
  print(name_end)
  val clean = "C:\\Users\\Tom\\Documents\\GitHub\\web2text\\src\\main\\resources\\cleaneval_style\\" + name_end
  val raw ="C:\\Users\\Tom\\Documents\\GitHub\\web2text\\src\\main\\resources\\raw_html\\" + name_end.dropRight(3) + "html"
  val a = Util.loadFile(raw)
  val b = Util.loadFile(clean)
  val c = Alignment.alignment(a, b)
  val out =  "C:\\Users\\Tom\\Documents\\GitHub\\web2text\\src\\main\\resources\\aligned\\" + name_end
  print(out)
  Util.save(out, c)

}
