package org.apache.spark.mllib.tests

import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.InfoThCriterionFactory
import org.apache.spark.mllib.feature.InfoThSelector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vector

object PruebaCFS {
    def main(args: Array[String]) {
      val conf = new SparkConf().setAppName("Prueba").setMaster("local")
      val sc=new SparkContext(conf)
      val criterion = new InfoThCriterionFactory("jmi")
      val nToSelect = 2
      val nPool = 0 // 0 -> w/o pool
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/car-mini.libsvm")
      val dataByFeature=data.flatMap({x => x.features.toArray.zipWithIndex.map({case (x,i) => (i, (x,x*x,1))})})//.map{ x => breeze.stats.stddev.apply(x.features.toBreeze) }
      val means=dataByFeature.reduceByKey({case ((s1,ss1,c1),(s2,ss2,c2)) => (s1+s2,ss1+ss2,c1+c2)})
                             .map({case (i,(sum, sumSquares, count)) =>
                                         val mean=sum/count
                                         //(i,mean*mean/count+sumSquares-2*mean*sum)})//Todas estas igualdades van sin Math.sqrt
                                         //(i,mean*sum+sumSquares-2*mean*sum)})
                                         //(i,mean*mean+sumSquares/count-2*mean*mean)})
                                         (i,math.sqrt(sumSquares/count-mean*mean))})
      println("STDDEVS:")
      means.foreach(println)
      val correlMatrix: Matrix = Statistics.corr(data.map({ x => Vectors.dense(x.features.toArray) }), "spearman")
      println(correlMatrix.toString())
    }
    
  }