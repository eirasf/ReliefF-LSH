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
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      val sc=new SparkContext(conf)
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/connect4small.libsvm")
      val numNeighbors=10;
//val data=fdata.sample(false, 0.01, 2)//sc.parallelize(fdata.take(5))

      val numElems=data.count().toDouble
      printf("NumElems: %f\n\n",numElems)
      val countsClass=data.map(_.label)
                          .countByValue
                          .map(
          {
            case (value, count) => (value,count/numElems)
          })
      printf("Class probs:\n-----------------\n")
      countsClass.toArray.sortBy(_._1).foreach({x => printf(" %f\t%f\n",x._1,x._2)})
      /*val countsAttributes=data.flatMap(_.features.toArray.zipWithIndex)
                                .map({case (value, index) => ((index, value), 1)})
                                .reduceByKey(_ + _)
                                .map({case ((index, value), count) => ((index, value),count/numElems)})
      countsAttributes.foreach(println)*/
      val rangeAttributes=data.flatMap(_.features.toArray.zipWithIndex) //Separate in (numAttribute,value) pairs
                            .map({case (value, index) => (index, (value, value))}) //Rearrange to compute max and min by reducing
                            .reduceByKey(//Compute max and min for each attribute index
                                {
                                  case ((max1, min1), (max2, min2)) =>
                                    (if (max1>max2) max1 else max2, if (min1<min2) min1 else min2)
                                })
                            .map(//Calculate range for each attribute
                                {
                                  //case (index, (max, min)) => (index, max-min)//Numeric
                                  case (index, (max, min)) => (index, 1) //Nominal
                                })
      
      val dCD=data.cartesian(data) //Will compare each instance with every other
                  .flatMap(//Remove comparisons between an instance and itself and compute distances
                  {
                     case (x,y) if (x==y) => None
                     case (x,y) if (x!=y) => Some(x, (y, x.features.toArray.zip(y.features.toArray).map(
                                                                                     //{x => math.abs(x._1-x._2)} //Numeric
                                                                                     {case (a,b) => if (a!=b) 1.0 else 0.0} //Nominal
                                                                                     ).sum))
                  })//.filter(_!=null)//By using flatMap and None/Some values this filter is avoided
            .groupByKey//Group by instance
            .map(//Group by class for each instance so that we can take K neighbors for each class for each instance
                {
                  case (x, nearestNeighborsByClass) => (x, nearestNeighborsByClass.groupBy({case (y, distances) => y.label}))
                })
            .map(//Sort by distance and select K neighbors for each group
                {
                  case(x, nearestNeighborsByClass) =>
                              (x,nearestNeighborsByClass.map(
                                  {
                                    case (y, distances) => (y,distances.toSeq.sortBy({case(y,d) => d}) //Sort by distance
                                                                                        .take(numNeighbors)) //Take the K nearest neighbors
                                   })
                                   .map({case(y,distances) => (y,distances.map({case(y,d) => (y,d,distances.length)}))}) //Add the number of neighbors so that we can divide later
                              )
                })
            .flatMap(//Ungroup everything in order to closer to addends
                {
                  case (x, nearestNeighborsByClass) =>
                    nearestNeighborsByClass.flatMap({y=>List(y._2)}).flatten.map({y => (x,y)})
                })
            .map(//Compute multipliers for each addend depending on their class
                {
                  case(x,(y,d,k)) if (x.label==y.label) => (x.features, y.features, -1.0/k)
                  case(x,(y,d,k)) if (x.label!=y.label) => (x.features, y.features, countsClass.get(y.label).get/((1.0-countsClass.get(x.label).get)*k))
                }
                )
//.foreach(println)
            .flatMap(//Separate everything into addends for each attribute, and rearrange so that the attribute index is the key 
                {
                  case(xFeat, yFeat, s) => xFeat.toArray.zip(yFeat.toArray).zipWithIndex.map(
                                                                             //{case ((x,y),i) => (math.abs(x-y), i)}).map({z => (z._2,z._1*s)}) //Numeric
                                                                             {case ((x,y),i) => (if (x!=y) 1 else 0, i)}).map({z => (z._2,z._1*s)}) //Nominal
                }
                )
//.foreach(println)
            .reduceByKey({_ + _})//Sum up for each attribute
            .join(rangeAttributes)//In order to divide by the range of each attribute
//.foreach(println)
            .map(//Add 1 to the attribNum so that is in the [1,N] range and divide each result by m and k.
                {
                  case(attribNum, (sum, range)) => (attribNum+1, sum/(range*numElems))
                })
      dCD.sortBy(_._2, false).foreach(
          {
            case (index, weight) => printf("Attribute %d: %f\n",index,weight)
          })
      //Tenemos (instance_original, Lista_de_vecinos(clase, Lista(distancias_vecinos)))
      //Interesa tener (numAtributo, List (factor, Lista(valor_A_I,sumando))) Donde factor es -1/mk si la clase es igual y PC/(1-PCRi) si es distinta
    }
  }