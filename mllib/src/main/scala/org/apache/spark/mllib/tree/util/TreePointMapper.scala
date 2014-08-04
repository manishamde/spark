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

package org.apache.spark.mllib.tree.util

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.Bin
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.point.TreePoint
import org.apache.spark.mllib.tree.configuration.Strategy

object TreePointMapper {

  def featuresToBinMap(
      lpRdd: RDD[LabeledPoint],
      bins: Array[Array[Bin]],
      strategy: Strategy): RDD[TreePoint] = {

    val numFeatures = lpRdd.take(1)(0).features.size

    val numBins = bins(0).length

    val categoricalFeaturesInfo = strategy.categoricalFeaturesInfo

    val isMulticlassClassification = strategy.isMulticlassClassification


    /**
     * Find bin for one (labeledPoint, feature).
     */
    def findBin(
                 featureIndex: Int,
                 labeledPoint: LabeledPoint,
                 isFeatureContinuous: Boolean,
                 isSpaceSufficientForAllCategoricalSplits: Boolean): Byte = {
      val binForFeatures = bins(featureIndex)
      val feature = labeledPoint.features(featureIndex)

      /**
       * Binary search helper method for continuous feature.
       */
      def binarySearchForBins(): Int = {
        var left = 0
        var right = binForFeatures.length - 1
        while (left <= right) {
          val mid = left + (right - left) / 2
          val bin = binForFeatures(mid)
          val lowThreshold = bin.lowSplit.threshold
          val highThreshold = bin.highSplit.threshold
          if ((lowThreshold < feature) && (highThreshold >= feature)) {
            return mid
          }
          else if (lowThreshold >= feature) {
            right = mid - 1
          }
          else {
            left = mid + 1
          }
        }
        -1
      }

      /**
       * Sequential search helper method to find bin for categorical feature in multiclass
       * classification. The category is returned since each category can belong to multiple
       * splits. The actual left/right child allocation per split is performed in the
       * sequential phase of the bin aggregate operation.
       */
      def sequentialBinSearchForUnorderedCategoricalFeatureInClassification(): Int = {
        labeledPoint.features(featureIndex).toInt
      }

      /**
       * Sequential search helper method to find bin for categorical feature
       * (for classification and regression).
       */
      def sequentialBinSearchForOrderedCategoricalFeature(): Int = {
        val featureCategories = strategy.categoricalFeaturesInfo(featureIndex)
        val featureValue = labeledPoint.features(featureIndex)
        var binIndex = 0
        while (binIndex < featureCategories) {
          val bin = bins(featureIndex)(binIndex)
          val categories = bin.highSplit.categories
          if (categories.contains(featureValue)) {
            return binIndex
          }
          binIndex += 1
        }
        if (featureValue < 0 || featureValue >= featureCategories) {
          throw new IllegalArgumentException(
            s"DecisionTree given invalid data:" +
              s" Feature $featureIndex is categorical with values in" +
              s" {0,...,${featureCategories - 1}," +
              s" but a data point gives it value $featureValue.\n" +
              "  Bad data point: " + labeledPoint.toString)
        }
        -1
      }

      if (isFeatureContinuous) {
        // Perform binary search for finding bin for continuous features.
        val binIndex = binarySearchForBins()
        if (binIndex == -1) {
          throw new UnknownError("no bin was found for continuous variable.")
        }
        binIndex.toByte
      } else {
        // Perform sequential search to find bin for categorical features.
        val binIndex = {
          val isUnorderedFeature =
            isMulticlassClassification && isSpaceSufficientForAllCategoricalSplits
          if (isUnorderedFeature) {
            sequentialBinSearchForUnorderedCategoricalFeatureInClassification()
          } else {
            sequentialBinSearchForOrderedCategoricalFeature()
          }
        }
        if (binIndex == -1) {
          throw new UnknownError("no bin was found for categorical variable.")
        }
        binIndex.toByte
      }
    }


    def lpToTp(labeledPoint: LabeledPoint): TreePoint = {

      val arr = new Array[Byte](numFeatures)
      var featureIndex = 0
      while (featureIndex < numFeatures) {
        val featureInfo = categoricalFeaturesInfo.get(featureIndex)
        val isFeatureContinuous = featureInfo.isEmpty
        if (isFeatureContinuous) {
          arr(featureIndex)
            = findBin(featureIndex, labeledPoint, isFeatureContinuous, false)
        } else {
          val featureCategories = featureInfo.get
          val isSpaceSufficientForAllCategoricalSplits
          = numBins > math.pow(2, featureCategories.toInt - 1) - 1
          arr(featureIndex)
            = findBin(featureIndex, labeledPoint, isFeatureContinuous,
            isSpaceSufficientForAllCategoricalSplits)
        }
        featureIndex += 1
      }
      new TreePoint(labeledPoint.label, arr)
    }

    lpRdd.map(lp => lpToTp(lp))

  }




}
