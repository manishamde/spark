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

import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.point.WeightedLabeledPoint
import org.apache.spark.mllib.model.Model
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.GradientBoostingModel
import org.apache.spark.mllib.tree.configuration.Algo._


class GradientBoost (private val strategy: Strategy) extends Serializable with Logging {

  def train(input: RDD[WeightedLabeledPoint]): Model = {
    strategy.algo match {
      // case Classification => GradientBoost.classification(input, strategy)
      case Regression => GradientBoost.regression(input, strategy)
    }
  }

}

object GradientBoost {

  def train(input: RDD[LabeledPoint], strategy: Strategy): Model = {
    val weightedInput = input.map(x => WeightedLabeledPoint(x.label, x.features))
    new AdaBoost(strategy).train(weightedInput)
  }

  private def regression(input: RDD[WeightedLabeledPoint], strategy: Strategy): Model = {

    // Initialize SAMME parameters
    val M = strategy.boostingIterations
    val K = strategy.numClassesForClassification
    val trees = new Array[Model](M + 1)

    // 1. Initialize tree
    trees(0) = new DecisionTree(strategy).train(input)

    var m = 1
    var data = input
    while (m <= M) {
      val model = new DecisionTree(strategy).train(data)
      // TODO: Perform line search for non-least squares calculations
      val gamma = 1
      trees(m) = model
      // TODO: Think about checkpointing for deeper iterations
      //update data with pseudo-residuals
      data = data.map(point => WeightedLabeledPoint(calculate(model, point), point.features))
      m += 1
    }

    // 3. Output classifier
    new GradientBoostingModel(trees)

  }

  // TODO: Implement Stochastic gradient boosting
  // TODO: Add learning rate

  def calculate(model: Model, point: WeightedLabeledPoint): Double = {
    point.label - model.predict(point.features)
  }

}
