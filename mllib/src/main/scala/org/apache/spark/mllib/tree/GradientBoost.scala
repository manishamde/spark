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

package org.apache.spark.mllib.tree

import org.apache.spark.SparkContext._

import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.point.WeightedLabeledPoint
import org.apache.spark.mllib.model.Model
import org.apache.spark.mllib.regression.{RegressionModel, LabeledPoint}
import org.apache.spark.mllib.tree.model.GradientBoostingModel
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.loss.{LeastAbsoluteError, LeastSquaresError}


class GradientBoost (private val strategy: Strategy) extends Serializable with Logging {

  def train(input: RDD[WeightedLabeledPoint]): Model = {
    strategy.algo match {
      // case Classification => GradientBoost.classification(input, strategy)
      case Regression => GradientBoost.regression(input, strategy)
    }
  }

}

object GradientBoost extends Logging {

  def train(input: RDD[LabeledPoint], strategy: Strategy): Model = {
    val weightedInput = input.map(x => WeightedLabeledPoint(x.label, x.features))
    new GradientBoost(strategy).train(weightedInput)
  }

  private def regression(input: RDD[WeightedLabeledPoint], strategy: Strategy): Model = {

    // Initialize gradient boosting parameters
    val M = strategy.boostingIterations
    val trees = new Array[Model](M + 1)
    // TODO: Add to strategy
    //val loss = new LeastSquaresError()
    val loss = new LeastAbsoluteError()

    // Cache input
    input.cache()

    logDebug("##########")
    logDebug("Building tree 0")
    logDebug("##########")
    var data = input

    // 1. Initialize tree
    val firstModel = new DecisionTree(strategy).train(data)
    trees(0) = firstModel
    logDebug("error of tree = " + meanSquaredError(firstModel, data))
    logDebug(data.first.toString)

    // psuedo-residual for second iteration
    data = data.map(point => WeightedLabeledPoint(loss.calculateResidual(firstModel, point),
      point.features))


    var m = 1
    while (m <= M) {
      logDebug("###################################################")
      logDebug("Gradient boosting tree iteration " + m)
      logDebug("###################################################")
      val model = new DecisionTree(strategy).train(data)
      trees(m) = model
      logDebug("error of tree = " + meanSquaredError(model, data))
      //update data with pseudo-residuals
      data = data.map(point => WeightedLabeledPoint(loss.calculateResidual(model, point),
        point.features))
      logDebug(data.first.toString)
      m += 1
    }

    // 3. Output classifier
    new GradientBoostingModel(trees)

  }

  // TODO: Port this method to a generic metrics package
  /**
   * Calculates the mean squared error for regression.
   */
  private def meanSquaredError(tree: Model, data: RDD[WeightedLabeledPoint]): Double = {
    data.map { y =>
      val err = tree.predict(y.features) - y.label
      err * err
    }.mean()
  }


  // TODO: Pluggable loss functions
  // TODO: Think about checkpointing for deeper iterations
  // TODO: Implement Stochastic gradient boosting
  // TODO: Add learning rate

}
