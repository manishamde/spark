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
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.point.WeightedLabeledPoint
import org.apache.spark.mllib.tree.model.{MultiClassAdaModel, DecisionTreeModel}
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.model.Model

class AdaBoost (private val strategy: Strategy) extends Serializable with Logging {

  def train(input: RDD[WeightedLabeledPoint]): Model = {
    strategy.algo match {
      case Classification => AdaBoost.classification(input, strategy)
      // case Regression => AdaBoost.regression(input, strategy)
    }
  }

}

object AdaBoost {

  def train(input: RDD[LabeledPoint], strategy: Strategy): Model = {
    val weightedInput = input.map(x => WeightedLabeledPoint(x.label, x.features))
    new AdaBoost(strategy).train(weightedInput)
  }

  private def classification(input: RDD[WeightedLabeledPoint], strategy: Strategy): Model = {

    // Initialize SAMME parameters
    val M = strategy.boostingIterations
    val K = strategy.numClassesForClassification
    val alphas = new Array[Double](M)
    val trees = new Array[Model](M)

    // SAMME
    var weightedInput = input
    // 2. For m = 1 to M:
    var m = 0
    while (m < M) {
      // (a) Fit a classifier T(m)(x) to the training data using weights wi.
      trees(m) = new DecisionTree(strategy).train(weightedInput)
      // (b) Compute err(m)
      val weightedTotalError
        = weightedInput.map(x => x.weight * unequalIdentity(trees(m).predict(x.features),
        x.label)).sum()
      val totalWeight = weightedInput.map(x => x.weight).sum()
      // TODO: Stop if the error is at least as bad as random guessing
      val err = weightedTotalError / totalWeight
      // (c) Compute alpha(m)
      alphas(m) = math.log((1- err)/err) + math.log(K - 1)
      // (d) Set weights
      // TODO: Ensure weights are non-negative
      weightedInput
        = weightedInput.map(x => WeightedLabeledPoint(x.label, x.features,
        x.weight * alphas(m) * unequalIdentity(trees(m).predict(x.features), x.label)))
      // (e) Renormalize weights
      val totalWeightAfterReweighing = weightedInput.map(x => x.weight).sum()
      weightedInput = weightedInput.map(x => WeightedLabeledPoint(x.label, x.features,
        x.weight / totalWeightAfterReweighing ))
      m += 1
    }

    // 3. Output classifier

    new MultiClassAdaModel(alphas, trees, strategy)

  }

  // TODO: SAMME.R
  // TODO: AdaBoost.Regression

  private def unequalIdentity(a: Double, b: Double): Double = {
    if (a != b) 1.0 else 0.0
  }

}
