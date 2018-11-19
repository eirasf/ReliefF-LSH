package org.apache.spark.mllib.feature

//import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

import es.udc.graph.LSHKNNGraphBuilder
import org.apache.spark.ml.classification.KNNFinder


object TestFeatureSelector
{
    def main(args: Array[String])
    {
      val options=parseParams(args)
      
      var file=options("dataset").asInstanceOf[String]
      
      var fileOut=options("output").asInstanceOf[String]
      
      //Set up Spark Context
      //val conf = new SparkConf().setAppName("PruebaReliefF").setMaster("local[8]") //DEBUG!!!!!!!!!!!!!!!!!!!!!!!
      //conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      conf.set("spark.eventLog.enabled", "true")
//      conf.set("spark.eventLog.dir","file:///home/eirasf/Escritorio/Tmp-work/sparklog-local")
      
      val spark = SparkSession.builder.appName("Simple Application")
                                    .master("local[1]")
                                    .getOrCreate()
      
      //val sc=new SparkContext(conf)
      val sc=spark.sparkContext
      sc.setLogLevel("WARN")//DEBUG!!!!!!!!!!!!!!!!!!!!!!!
      
      //Load data from file
      //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/LargeDatasets/libsvm/isoletTrain.libsvm")
      //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, file, -1, 3*sc.defaultParallelism)
      //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/car-mini.libsvm")
      
      //Set maximum number of near neighbors to be taken into account for each instance
      val numNeighbors=options("num_neighbors").asInstanceOf[Double].toInt

      //Set the type (numeric/discrete) for each attribute and class
     /* val attributeTypes=if (options.contains("attribute_types"))
                          options("attribute_types").asInstanceOf[String].toCharArray().map({case c => c=='N'})
                         else
                          (0 to data.first().features.size-1).map({case c => true}).toArray
      if (attributeTypes.length!=data.first().features.size)
      {
        println("The number of features ("+attributeTypes.length+") does not match the instances in the file ("+data.first().features.size+")")
        return
      }
      */
      //FIX in order to have .toDF()
      val sqlContext= new org.apache.spark.sql.SQLContext(sc)
      import sqlContext.implicits._
      
      var discreteClass=(options.contains("class_type")==false) || (options("class_type").asInstanceOf[String].toLowerCase()!="n")
      
      val method=options("method").asInstanceOf[String]
      
      val training = MLUtils.loadLibSVMFile(sc, "file:///mnt/NTFS/owncloud/LargeDatasets/libsvm/car.libsvm").toDF()
      val trainingML = MLUtils.convertVectorColumnsToML(training)
      
      val knn = new KNNFinder()
        .setTopTreeSize(trainingML.count().toInt / 500)
        .setK(numNeighbors)
      
      val knnModel = knn.fitFinder(trainingML.toDF())
      
      println(knnModel.topTree)
      
      val original=trainingML.sort($"features").head(50)
      val neighbors=knnModel.find(trainingML.toDF()).take(50)
      
      for ((e,n) <- neighbors)
      {
        println("ORIGINAL:"+e)
        for (ng <- n)
          println("\t"+ng)
      }
      
      
      
      //Stop the Spark Context
      sc.stop()
    }
    
    def showUsageAndExit()=
    {
      println("""Usage: ReliefFFeatureSelector dataset [options]
        Dataset must be a libsvm or text file
    Options:
        -t    Attribute types. String consisting of N or C for each attribute
        -ct    Class type. Either N (numerical) or C (categorical)  
        -k    Number of neighbors (default: """+ReliefFFeatureSelector.DEFAULT_K+""")
        -m    Method used to compute the graph. Valid values: lsh, brute (default: """+ReliefFFeatureSelector.DEFAULT_METHOD+""")
        -r    Starting radius (default: """+LSHKNNGraphBuilder.DEFAULT_RADIUS_START+""")
        -c    Maximum comparisons per item (default: auto)
    
    Advanced LSH options:
        -n    Number of hashes per item (default: auto)
        -l    Hash length (default: auto)""")
      System.exit(-1)
    }
    
    def parseParams(p:Array[String]):Map[String, Any]=
    {
      val m=scala.collection.mutable.Map[String, Any]("num_neighbors" -> ReliefFFeatureSelector.DEFAULT_K.toDouble,
                                                      "method" -> ReliefFFeatureSelector.DEFAULT_METHOD,
                                                      "radius_start" -> LSHKNNGraphBuilder.DEFAULT_RADIUS_START)
      if (p.length<=0)
        showUsageAndExit()
      
      m("dataset")=p(0)
      m("output")=p(0).substring(0,p(0).lastIndexOf("."))+"-out.txt"
      
      var i=1
      while (i < p.length)
      {
        if ((i>=p.length-1) || (p(i).charAt(0)!='-'))
        {
          println("Unknown option: "+p(i))
          showUsageAndExit()
        }
        val readOptionName=p(i).substring(1)
        val option=readOptionName match
          {
            case "k"   => "num_neighbors"
            case "m"   => "method"
            case "t"   => "attribute_types"
            case "ct"   => "class_type"
            case "r"   => "radius_start"
            case "n"   => "num_tables"
            case "l"   => "key_length"
            case "c"   => "max_comparisons"
            case "o"   => "output"
            case somethingElse => readOptionName
          }
        if (!m.keySet.exists(_==option) && option==readOptionName)
        {
          println("Unknown option:"+readOptionName)
          showUsageAndExit()
        }
        if (option=="method")
        {
          if (p(i+1)=="lsh" || p(i+1)=="brute")
            m(option)=p(i+1)
          else
          {
            println("Unknown method:"+p(i+1))
            showUsageAndExit()
          }
        }
        else
        {
          if ((option=="class_type") || (option=="attribute_types") || (option=="output"))
            m(option)=p(i+1)
          else
            m(option)=p(i+1).toDouble
        }
          
        i=i+2
      }
      return m.toMap
    }
  }