/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tree.model

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.EnsembleCombiningStrategy._
import org.apache.spark.rdd.RDD

import scala.collection.mutable

@Experimental
class WeightedEnsembleModel(
    val baseLearners: Array[DecisionTreeModel],
    baseLearnerWeights: Array[Double],
    algo: Algo,
    combiningStrategy: EnsembleCombiningStrategy) extends Serializable {

  require(numTrees > 0, s"WeightedEnsembleModel cannot be created without base learners. Number " +
    s"of baselearners = $baseLearners")

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features array representing a single data point
   * @return predicted category from the trained model
   */
  private def predictRaw(features: Vector): Double = {
    val treePredictions = baseLearners.map(learner => learner.predict(features))
    if (numTrees == 1){
      treePredictions(0)
    } else {
      var prediction = treePredictions(0)
      var index = 1
      while (index < numTrees) {
        prediction += baseLearnerWeights(index) * treePredictions(index)
        index += 1
      }
      prediction
    }
  }

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features array representing a single data point
   * @return predicted category from the trained model
   */
  private def predictBySumming(features: Vector): Double = {
    val treePredictions = baseLearners.map(learner => learner.predict(features))
    val rawPrediction = {
      if (numTrees == 1) {
        treePredictions(0)
      } else {
        var prediction = treePredictions(0)
        var index = 1
        while (index < numTrees) {
          prediction += baseLearnerWeights(index) * treePredictions(index)
          index += 1
        }
        prediction
      }
    }
    algo match {
      case Regression => predictRaw(features)
      case Classification => if (predictRaw(features) > 0 ) 1.0 else 0.0
      case _ => throw new IllegalArgumentException(
        s"WeightedEnsembleModel given unknown algo parameter: $algo.")
    }
  }

  /**
   * Predict values for a single data point.
   *
   * @param features array representing a single data point
   * @return Double prediction from the trained model
   */
  def predictByAveraging(features: Vector): Double = {
    algo match {
      case Classification =>
        val predictionToCount = new mutable.HashMap[Int, Int]()
        baseLearners.foreach { learner =>
          val prediction = learner.predict(features).toInt
          predictionToCount(prediction) = predictionToCount.getOrElse(prediction, 0) + 1
        }
        predictionToCount.maxBy(_._2)._1
      case Regression =>
        baseLearners.map(_.predict(features)).sum / baseLearners.size
    }
  }


  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features array representing a single data point
   * @return predicted category from the trained model
   */
  def predict(features: Vector): Double = {
    combiningStrategy match {
      case Sum => predictBySumming(features)
      case Average => predictByAveraging(features)
      case _ => throw new IllegalArgumentException(
        s"WeightedEnsembleModel given unknown combining parameter: $combiningStrategy.")
    }
  }

  /**
   * Predict values for the given data set.
   *
   * @param features RDD representing data points to be predicted
   * @return RDD[Double] where each entry contains the corresponding prediction
   */
  def predict(features: RDD[Vector]): RDD[Double] = features.map(x => predict(x))

  /**
   * Print full model.
   */
  override def toString: String = {
    val header = algo match {
      case Classification =>
        s"WeightedEnsembleModel classifier with $numTrees trees\n"
      case Regression =>
        s"WeightedEnsembleModel regressor with $numTrees trees\n"
      case _ => throw new IllegalArgumentException(
        s"WeightedEnsembleModel given unknown algo parameter: $algo.")
    }
    header + baseLearners.zipWithIndex.map { case (learner, treeIndex) =>
      s"  Tree $treeIndex:\n" + learner.topNode.subtreeToString(4)
    }.fold("")(_ + _)
  }

  /**
   * Print the full model to a string.
   */
  def toDebugString: String = {
    val header = toString + "\n"
    header + baseLearners.zipWithIndex.map { case (tree, treeIndex) =>
      s"  Tree $treeIndex:\n" + tree.topNode.subtreeToString(4)
    }.fold("")(_ + _)
  }


  // TODO: Remove these helpers methods once class is generalized to support any base learning
  // algorithms.

  /**
   * Get number of trees in forest.
   */
  def numTrees: Int = baseLearners.size

  /**
   * Get total number of nodes, summed over all trees in the forest.
   */
  def totalNumNodes: Int = baseLearners.map(tree => tree.numNodes).sum


}