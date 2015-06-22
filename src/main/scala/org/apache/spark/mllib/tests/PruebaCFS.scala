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
import org.apache.spark.broadcast.Broadcast

object PruebaCFS {
    
    def evalSubset(attribIndices: Array[Int], devs: Array[Double], bCorrelMatrix: Broadcast[Matrix]) : Double =
    {
      val classIndex=bCorrelMatrix.value.numCols-1;
      val fraction=attribIndices.map({x => (devs(x) * bCorrelMatrix.value(x, classIndex),
                                            attribIndices.map({case a  if (a!=x) => 2.0 * devs(x) * devs(a) * bCorrelMatrix.value(x, a)
                                                                case a => 0.0})
                                                          .reduce(_+_)
                                                          + devs(x) * devs(x))})
                   .reduce({(a,b) => (a._1+b._1, a._2+b._2)})
      println(fraction)
      if (fraction._2==0.0)
        return 0.0
      return fraction._1/math.sqrt(fraction._2)
    }
  
    def main(args: Array[String]) {
      val conf = new SparkConf().setAppName("Prueba").setMaster("local")
      val sc=new SparkContext(conf)
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/car-mini.libsvm")
      
      
      
      
      
      
      
      
      val dataByFeature=data.flatMap({x => x.features.toArray.view.zipWithIndex.map({case (x,i) => (i, (x,x*x,1))}) :+ (Int.MaxValue,(x.label, x.label*x.label, 1))})
      dataByFeature.foreach(println)
      val stddevs=dataByFeature.reduceByKey({case ((s1,ss1,c1),(s2,ss2,c2)) => (s1+s2,ss1+ss2,c1+c2)})
                             .map({case (i,(sum, sumSquares, count)) =>
                                         val mean=sum/count
                                         //(i,mean*mean/count+sumSquares-2*mean*sum)})//Todas estas igualdades van sin Math.sqrt
                                         //(i,mean*sum+sumSquares-2*mean*sum)})
                                         //(i,mean*mean+sumSquares/count-2*mean*mean)})
                                         (i,math.sqrt(sumSquares/count-mean*mean))})
                             .sortByKey()
                             .map(_._2)
      println("STDDEVS:")
      stddevs.foreach(println)
      val correlMatrix: Matrix = Statistics.corr(data.map({ x => Vectors.dense(x.features.toArray :+ x.label) }), "pearson")//"spearman")
                                           .map(math.abs(_))
println(correlMatrix)      
      val devs=stddevs.collect()
      val bCorrelMatrix=sc.broadcast(correlMatrix)

      val attribIndices=new Array[Int](1)
      attribIndices(0)=0
      
      val merit=evalSubset(attribIndices, devs, bCorrelMatrix)
      println("Merit: "+merit)
    }
  }