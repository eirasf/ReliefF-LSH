package org.apache.spark.ml.classification

import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.knn.Tree
import org.apache.spark.ml.linalg.Vectors
import scala.collection.mutable.ArrayBuffer

class KNNFinder extends KNNClassifier
{
  def fitFinder(dataset: Dataset[_]): KNNFinderModel =
  {
    val m=fit(dataset)
    KNNFinderModel.fromClassificationModel(m)
  }
  
}

class KNNFinderModel private[ml](
                                          override val uid: String,
                                          override val topTree: Broadcast[Tree],
                                          override val subTrees: RDD[Tree],
                                          override val _numClasses: Int
                                        ) extends KNNClassificationModel(uid, topTree, subTrees, _numClasses)
{
  def find(dataset: Dataset[_]):RDD[(Row, Array[(Row, Double)])]  = {
    val neighborRDD : RDD[(Long, Array[(Row, Double)])] = transform(dataset, topTree, subTrees)
    
    dataset.toDF().rdd.zipWithIndex().map { case (row, i) => (i, row) }
        .leftOuterJoin(neighborRDD)
        .map({case (i, (element, neighbors)) => (element, neighbors.get)})
  }
}

object KNNFinderModel
{
  def fromClassificationModel(classificationModel:KNNClassificationModel):KNNFinderModel = 
  {
    val m=new KNNFinderModel(classificationModel.uid, classificationModel.topTree, classificationModel.subTrees, classificationModel._numClasses)
    //m.setK(m.$(classificationModel.k))
    m.setK(classificationModel.getOrDefault(classificationModel.k))
    
    //m.setBufferSize(m.$(classificationModel.bufferSize))
    m.setBufferSize(classificationModel.getOrDefault(classificationModel.bufferSize))
    m
  }
}