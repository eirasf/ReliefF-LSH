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
                       
      val dCD=data.cartesian(data).flatMap(
          {
             case (x,y) if (x==y) => None
             case (x,y) if (x!=y) => Some(x, (y, x.features.toArray.zip(y.features.toArray).map(
                                                                             {x => math.abs(x._1-x._2)}
                                                                             ).sum))
          })//.filter(_!=null)//By using flatMap and None/Some values this filter is avoided.
            .groupByKey
            .map(
                {
                  case (x, nearestNeighborsByClass) => (x, nearestNeighborsByClass.groupBy({case (y, distances) => y.label}))
                })
            .map(
                {
                  case(x, nearestNeighborsByClass) =>
                              (x,nearestNeighborsByClass.map(
                                  {
                                    case (y, distances) => (y,distances.toSeq.sortBy({case(y,d) => d}) //Sort by distance
                                                                                        .take(5) //Take the 5 nearest neighbors
                                                                                        .map({case(y,d) => (y,d,5)})) //En vez de poner aquí 5 a pincho, poner el número de vecinos que efectivamente hay (que habrá que hacerlo más adelante)
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                   })
                              )
                })
            .flatMap(
                {
                  case (x, nearestNeighborsByClass) =>
                    nearestNeighborsByClass.flatMap({y=>List(y._2)}).flatten.map({y => (x,y)})
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