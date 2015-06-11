package org.apache.spark.mllib.tests

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf

object PruebaReliefFNumeric {
    def main(args: Array[String]) {
      val conf = new SparkConf().setAppName("PruebaReliefF").setMaster("local")
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      val sc=new SparkContext(conf)
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/car-mini.libsvm")
      val numNeighbors=10;
//val data=fdata.sample(false, 0.01, 2)//sc.parallelize(fdata.take(5))
      val numberedData=data.zipWithIndex().map({case x=>(x._2.toInt,x._1)})
      val numElems=data.count().toDouble
      printf("NumElems: %f\n\n",numElems)
val bnData=sc.broadcast(numberedData.collect().toMap)
var indices=sc.parallelize(0 to numElems.toInt-1)
var cart=indices.cartesian(indices)
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
      val maxMinClass=data.map(
                            {case x => (x.label, x.label)}
                            )
                         .reduce(//Compute max and min
                            {
                              case ((max1, min1), (max2, min2)) =>
                                (if (max1>max2) max1 else max2, if (min1<min2) min1 else min2)
                            })
      val rangeClass=maxMinClass._1-maxMinClass._2
printf("\n\nRange Class:"+rangeClass+"\n")
      val rangeAttributes=data.flatMap(_.features.toArray.zipWithIndex) //Separate in (numAttribute,value) pairs
                            .map({case (value, index) => (index, (value, value))}) //Rearrange to compute max and min by reducing
                            .reduceByKey(//Compute max and min for each attribute index
                                {
                                  case ((max1, min1), (max2, min2)) =>
                                    (if (max1>max2) max1 else max2, if (min1<min2) min1 else min2)
                                })
                            .map(//Calculate range for each attribute
                                {
                                  case (index, (max, min)) => (index, max-min)//Numeric
                                  //case (index, (max, min)) => (index, 1) //Nominal
                                })
printf("\n\nRange attributes:\n-----------------\n")
rangeAttributes.foreach(println)
printf("\n\nAddends:\n-----------------\n")
      /*val dCD=data.cartesian(data) //Will compare each instance with every other
                  .flatMap(//Remove comparisons between an instance and itself and compute distances
                  {
                     case (x,y) if (x==y) => None
                     case (x,y) if (x!=y) => Some(x, (y, x.features.toArray.zip(y.features.toArray).map(
                                                                                     //{x => math.abs(x._1-x._2)} //Numeric
                                                                                     {case (a,b) => if (a!=b) 1.0 else 0.0} //Nominal
                                                                                     ).sum))
                  })//.filter(_!=null)//By using flatMap and None/Some values this filter is avoided*/
val dCD=cart //Will compare each instance with every other
                  .flatMap(//Remove comparisons between an instance and itself and compute distances
                  {
                     case (x,y) if (x==y) => None
                     case (x,y) if (x!=y) => Some(x, (y, bnData.value.get(x).get.features.toArray.zip(bnData.value.get(y).get.features.toArray).map(
                                                                                     {x => math.abs(x._1-x._2)} //Numeric
                                                                                     //{case (a,b) => if (a!=b) 1.0 else 0.0} //Nominal
                                                                                     ).sum))
                  })//.filter(_!=null)//By using flatMap and None/Some values this filter is avoided
            .groupByKey//Group by instance
            /*.map(//Group by class for each instance so that we can take K neighbors for each class for each instance
                {
                  case (x, nearestNeighborsByClass) => (x, nearestNeighborsByClass.groupBy({case (y, distances) => bnData.value.get(y).get.label}))
                })*/
            .map(//Sort by distance and select K neighbors for each group
                {
                  case(x, distances) =>
                              (x,distances.toSeq.sortBy({case(y,d) => d}) //Sort by distance
                                          .take(numNeighbors)) //Take the K nearest neighbors
                })
            .map(
                {
                  case(x,distances) =>
                              (x,distances.map({case(y,d) => (y,distances.length)})) //Add the number of neighbors so that we can divide later
                })
            .flatMap(//Ungroup everything in order to closer to addends
                {
                  case (x, distances) =>
                      distances.flatMap({y => List((x,y))})
                })
//Usar este cómputo para calcular m_ndc y (m_nda,m_ndcda) por separado

val m_ndc=dCD.map(
                {
                  case (x, (y,k)) => (math.abs(bnData.value.get(x).get.label-bnData.value.get(y).get.label))
                }) //Will be normalized when computing the weight
                .reduce(_+_)
val numNeighborsObtained=dCD.first()._2._2
println("\nNeighbors obtained ="+numNeighborsObtained+"\n----------------")
println("\nm_ndc="+m_ndc+"\n----------------") //Falta dividir por numNeighborsObtained
/*            .map(//Compute multipliers for each addend depending on their class
                {
                  case(x,(y,k)) if (bnData.value.get(x).get.label==bnData.value.get(y).get.label) => (x, y, -1.0/k)
                  case(x,(y,k)) if (bnData.value.get(x).get.label!=bnData.value.get(y).get.label) => (x, y, countsClass.get(bnData.value.get(y).get.label).get/((1.0-countsClass.get(bnData.value.get(x).get.label).get)*k))
                }
                )
//.foreach(println)
*/
val weights=dCD.flatMap(//Separate everything into addends for each attribute, and rearrange so that the attribute index is the key 
                {
                  case(x, (y, k)) => bnData.value.get(x).get.features.toArray.zip(bnData.value.get(y).get.features.toArray).zipWithIndex.map(
                                                                             {
                                                                               case ((a,b),i) => (i,(math.abs(a-b), math.abs(a-b)*(math.abs(bnData.value.get(x).get.label-bnData.value.get(y).get.label))))})//Numeric
                                                                               //case ((a,b),i) if (i==0) => (x,y,i,math.abs(bnData.value.get(x).get.label-bnData.value.get(y).get.label),(math.abs(a-b), math.abs(a-b)*(math.abs(bnData.value.get(x).get.label-bnData.value.get(y).get.label))))//Numeric
                                                                               //case ((a,b),i) => (x,y,i,0,(math.abs(a-b), math.abs(a-b)*(math.abs(bnData.value.get(x).get.label-bnData.value.get(y).get.label))))})//Numeric
                                                                               //{case ((fx,fy),i) => (x,y,i, (if (fx!=fy) 1 else 0, if (fx!=fy) math.abs(bnData.value.get(x).get.label-bnData.value.get(y).get.label) else 0))})//Nominal
                })
//.filter(_._1==4)
//.foreach(println)
            .reduceByKey({case ((m_nda1, m_ndcda1), (m_nda2, m_ndcda2)) => (m_nda1 + m_nda2, m_ndcda1 + m_ndcda2)})//Sum up for each attribute
            .join(rangeAttributes)//In order to divide by the range of each attribute
//.foreach(println)
            .map(//Add 1 to the attribNum so that is in the [1,N] range and compute weights.
                {
                  case(attribNum, ((m_nda, m_ndcda),range)) => (attribNum+1, (m_ndcda/m_ndc - ((rangeClass*m_nda - m_ndcda)/(numNeighborsObtained*rangeClass*numElems-m_ndc)))/range)
                })
      weights.sortBy(_._2, false).foreach(
          {
            case (index, weight) => printf("Attribute %d: %f\n",index,weight)
          })
      //Tenemos (instance_original, Lista_de_vecinos(clase, Lista(distancias_vecinos)))
      //Interesa tener (numAtributo, List (factor, Lista(valor_A_I,sumando))) Donde factor es -1/mk si la clase es igual y PC/(1-PCRi) si es distinta
    }
  }