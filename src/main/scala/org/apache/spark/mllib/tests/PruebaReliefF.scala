package org.apache.spark.mllib.tests

import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.InfoThCriterionFactory
import org.apache.spark.mllib.feature.InfoThSelector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf

object PruebaReliefF {
    def main(args: Array[String]) {
      val conf = new SparkConf().setAppName("PruebaReliefF").setMaster("local")
      val sc=new SparkContext(conf)
      val fdata: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/car.libsvm")
      
val data=fdata.sample(false, 0.01, 2)//sc.parallelize(fdata.take(5))
      
      val dCD=data.cartesian(data).map(
          {
             case (x,y) => (x, y.label, x.features.toArray.zip(y.features.toArray).map(
                                                                             {x => math.abs(x._1-x._2)}
                                                                             ).sum)
          }).groupBy({case (x,yClass,d) => x})
            .map(
                {
                  case (x, nearestNeighborsByClass) => (x, nearestNeighborsByClass.groupBy({case (x, cl, distances) => cl}))
                })
            .map(
                {
                  case(x, nearestNeighborsByClass) => (x.label,nearestNeighborsByClass.map(
                                                                  {
                                                                    case (yClass, distances) => (yClass,distances.toSeq.sortBy({case(x,yClass,d) => d}) //Sort by distance
                                                                                                                        .take(5) //Take the 5 nearest neighbors
                                                                                                                        .map({case(x,yClass,d) => d})) //Get rid of everything but the distance
                                                                   }))
                })
      dCD.foreach(println)
    }
  }