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
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/car.libsvm")
      
//val data=fdata.sample(false, 0.01, 2)//sc.parallelize(fdata.take(5))

      val numElems=data.count().toDouble
      val countsClass=data.map(_.label)
                          .countByValue
                          .map(
          {
            case (value, count) => (value,count/numElems)
          })
      /*val countsAttributes=data.flatMap(_.features.toArray.zipWithIndex)
                                .map({case (value, index) => ((index, value), 1)})
                                .reduceByKey(_ + _)
                                .map({case ((index, value), count) => ((index, value),count/numElems)})
      countsAttributes.foreach(println)*/
      val rangeAttributes=data.flatMap(_.features.toArray.zipWithIndex)
                            .map({case (value, index) => (index, (value, value))})
                            .reduceByKey(
                                {
                                  case ((max1, min1), (max2, min2)) =>
                                    (if (max1>max2) max1 else max2, if (min1<min2) min1 else min2)
                                })
                            .map(
                                {
                                  case (index, (max, min)) => (index, max-min)
                                })
      
      val dCD=data.cartesian(data).flatMap(
          {
             case (x,y) if (x==y) => None
             case (x,y) if (x!=y) => Some(x, (y, x.features.toArray.zip(y.features.toArray).map(
                                                                             //{x => math.abs(x._1-x._2)} //Numeric
                                                                               {x => if (x==y) 1.0 else 0.0} //Nominal
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
                                                                                        .take(10)) //Take the 10 nearest neighbors
                                   })
                                   .map({case(y,distances) => (y,distances.map({case(y,d) => (y,d,distances.length)}))})
                              )
                })
            .flatMap(
                {
                  case (x, nearestNeighborsByClass) =>
                    nearestNeighborsByClass.flatMap({y=>List(y._2)}).flatten.map({y => (x,y)})
                })
            .map(
                {
                  case(x,(y,d,k)) if (x.label==y.label) => (x.features, y.features, -1.0/k)
                  case(x,(y,d,k)) if (x.label!=y.label) => (x.features, y.features, countsClass.get(y.label).get/(1.0-countsClass.get(x.label).get))
                }
                )
            .flatMap(
                {
                  case(xFeat, yFeat, s) => xFeat.toArray.zip(yFeat.toArray).zipWithIndex.map(
                                                                             //{case ((x,y),i) => (math.abs(x-y), i)}).map({z => (z._2,z._1*s)}) //Numeric
                                                                             {case ((x,y),i) => (if (x==y) 1 else 0, i)}).map({z => (z._2,z._1*s)}) //Nominal
                }
                )
            /*.mapValues(
                {
                  x => x/numElems
                })*/
            .reduceByKey({_ + _})
            //Falta dividir por el rango de cada atributo y por el numero de elementos
            .join(rangeAttributes)
            .map(
                {
                  case(attribNum, (sum, range)) => (attribNum+1, sum/(range*numElems))
                })
      dCD.sortBy(_._2, false).foreach(println)
      //Tenemos (instance_original, Lista_de_vecinos(clase, Lista(distancias_vecinos)))
      //Interesa tener (numAtributo, List (factor, Lista(valor_A_I,sumando))) Donde factor es -1/mk si la clase es igual y PC/(1-PCRi) si es distinta
    }
  }