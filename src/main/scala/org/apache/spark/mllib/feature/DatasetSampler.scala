package org.apache.spark.mllib.feature

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import java.io.PrintWriter
import java.io.File

object DatasetSampler
{  
    def main(args: Array[String])
    {
      if (args.length <= 2)
      {
        println("Usage: DatasetSampler fileIn sampleSizes numSamples")
        return
      }
      
      var file=args(0)
      
      var fileOut=file.substring(0,file.lastIndexOf("."))+"-sample"
      
      //Set up Spark Context
      val conf = new SparkConf().setAppName("DatasetSampler")//.setMaster("local[8]")
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      conf.set("spark.eventLog.enabled", "true")
//      conf.set("spark.eventLog.dir","file:///home/eirasf/Escritorio/Tmp-work/sparklog-local")
      
      val sc=new SparkContext(conf)
      
      //Load data from file
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, file)
      
      val sampleSizes=args(1).split("-")
      
      for (size <- sampleSizes)
      {
        var i=0
        for (i <- 1 to args(2).toInt)
        {
          val fileName=fileOut+size.toInt+"-"+i+".libsvm"
          println("Saving sample"+size+" "+i+" to "+fileName)
          MLUtils.saveAsLibSVMFile(sc.parallelize(data.takeSample(false, size.toInt, System.nanoTime()), 8), fileName)
        }
      }
      
      //Stop the Spark Context
      sc.stop()
    }
  }