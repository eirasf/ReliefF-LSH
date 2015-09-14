package org.apache.spark.mllib.tests

import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.InfoThCriterionFactory
import org.apache.spark.mllib.feature.InfoThSelector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf

object HelloWorld {
    def main(args: Array[String]) {
      val conf = new SparkConf().setAppName("Prueba").setMaster("local")
      val sc=new SparkContext(conf)
      val criterion = new InfoThCriterionFactory("mim")
      val nToSelect = 2
      val nPool = 0 // 0 -> w/o pool
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/car.libsvm")
      
      println("*** FS criterion: " + criterion.getCriterion.toString)
      println("*** Number of features to select: " + nToSelect)
      println("*** Pool size: " + nPool)
      
      val featureSelector = InfoThSelector.train(criterion, 
              data, // RDD[LabeledPoint]
              nToSelect, // number of features to select
            nPool) // number of features in pool
          featureSelector
      
      val reduced = data.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
      reduced.first()
    }
  }