package org.apache.spark.mllib.tests

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
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

      val numElems=data.count().toDouble
      val countsClass=data.map(_.label)
                          .countByValue
                          .map(
          {
            case (value, count) => (value,count/numElems)
          })
      val countsAttributes=data.flatMap(_.features.toArray.zipWithIndex)
                                .map({case (value, index) => ((index, value), 1)})
                                .reduceByKey(_ + _)
                                .map({case ((index, value), count) => ((index, value),count/numElems)})
                       
      val dCD=data.cartesian(data).map(
          {
             case (x,y) if (x==y) => null
             case (x,y) if (x!=y) => (x, (y.label, x.features.toArray.zip(y.features.toArray).map(
                                                                             {x => math.abs(x._1-x._2)}
                                                                             ).sum))
          }).filter(_!=null)
            .groupByKey
            .map(
                {
                  case (x, nearestNeighborsByClass) => (x, nearestNeighborsByClass.groupBy({case (cl, distances) => cl}))
                })
            .map(
                {
                  case(x, nearestNeighborsByClass) =>
                              (x,nearestNeighborsByClass.map(
                                {
                                  case (yClass, distances) => (yClass,distances.toSeq.sortBy({case(yClass,d) => d}) //Sort by distance
                                                                                      .take(5) //Take the 5 nearest neighbors
                                                                                      .map({case(yClass,d) => d})) //Get rid of everything but the distance
                                 }))
                })
      //Tenemos (instance_original, Lista_de_vecinos(clase, Lista(distancias_vecinos)))
      //Interesa tener (numAtributo, List (factor, Lista(valor_A_I,sumando))) Donde factor es -1/mk si la clase es igual y PC/(1-PCRi) si es distinta
      var bdCD=sc.broadcast(dCD.collect);
      var attributes=data.first.features.toArray.zipWithIndex
      var weights=attributes.map(
          {
            _._1
          })
      //weights.foreach(println)
      dCD.foreach(println)
    }
  }