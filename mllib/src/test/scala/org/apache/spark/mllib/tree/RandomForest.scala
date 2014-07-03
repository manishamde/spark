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

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.configuration.Algo.Algo
import org.apache.spark.mllib.model.Model
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.mllib.point.WeightedLabeledPoint

object RandomForest {

  def train(input: RDD[LabeledPoint],
      algo: Algo,
      impurity: Impurity,
      maxDepth: Int,
      numTrees: Int): Model = {
    val strategy
      = new Strategy(algo = algo, impurity = impurity, maxDepth = maxDepth, numTrees = numTrees,
      isRandomForest = true)
    val weightedInput = input.map(x => WeightedLabeledPoint(x.label, x.features))
    new DecisionTree(strategy).train(weightedInput)
  }

}
