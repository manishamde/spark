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

package org.apache.spark.mllib.tree.impl

import org.apache.spark.mllib.tree.impurity._

import scala.collection.mutable


/**
 * :: Experimental ::
 * DecisionTree statistics aggregator.
 * This holds a flat array of statistics for a set of (nodes, features, bins)
 * and helps with indexing.
 * TODO: Allow views of Vector types to replace some of the code in here.
 */
private[tree] class DTStatsAggregator(
    metadata: DecisionTreeMetadata,
    val numNodes: Int) extends Serializable {

  val impurityAggregator: ImpurityAggregator = metadata.impurity match {
    case Gini => new GiniAggregator(metadata.numClasses)
    case Entropy => new EntropyAggregator(metadata.numClasses)
    case Variance => new VarianceAggregator()
    case _ => throw new IllegalArgumentException(s"Bad impurity parameter: ${metadata.impurity}")
  }

  val statsSize: Int = impurityAggregator.statsSize

  val numFeatures: Int = metadata.numFeatures

  val numBins: Array[Int] = metadata.numBins

  val isUnordered: Array[Boolean] =
    Range(0, numFeatures).map(f => metadata.unorderedFeatures.contains(f)).toArray

  private val featureOffsets: Array[Int] = {
    def featureOffsetsCalc(total: Int, featureIndex: Int): Int = {
      if (isUnordered(featureIndex)) {
        total + 2 * numBins(featureIndex)
      } else {
        total + numBins(featureIndex)
      }
    }
    Range(0, numFeatures).scanLeft(0)(featureOffsetsCalc).map(statsSize * _).toArray
  }

  /**
   * Number of elements for each node, corresponding to stride between nodes in [[allStats]].
   */
  private val nodeStride: Int = featureOffsets.last * statsSize

  /**
   * Total number of elements stored in this aggregator.
   */
  val allStatsSize: Int = numNodes * nodeStride

  /**
   * Flat array of elements.
   * Index for start of stats for a (node, feature, bin) is:
   *   index = nodeIndex * nodeStride + featureOffsets(featureIndex) + binIndex * statsSize
   * Note: For unordered features, the left child stats have binIndex in [0, numBins(featureIndex))
   *       and the right child stats in [numBins(featureIndex), 2 * numBins(featureIndex))
   */
  val allStats: Array[Double] = new Array[Double](allStatsSize)

  /**
   * Get an [[ImpurityCalculator]] for a given (node, feature, bin).
   * @param nodeFeatureOffset  For ordered features, this is a pre-computed (node, feature) offset
   *                           from [[getNodeFeatureOffset]].
   *                           For unordered features, this is a pre-computed
   *                           (node, feature, left/right child) offset from
   *                           [[getLeftRightNodeFeatureOffsets]].
   */
  def getImpurityCalculator(nodeFeatureOffset: Int, binIndex: Int): ImpurityCalculator = {
    impurityAggregator.getCalculator(allStats, nodeFeatureOffset + binIndex * statsSize)
  }

  /**
   * Update the stats for a given (node, feature, bin) for ordered features, using the given label.
   */
  def update(nodeIndex: Int, featureIndex: Int, binIndex: Int, label: Double): Unit = {
    val i = nodeIndex * nodeStride + featureOffsets(featureIndex) + binIndex * statsSize
    impurityAggregator.update(allStats, i, label)
  }

  /**
   * Pre-compute node offset for use with [[nodeUpdate]].
   */
  def getNodeOffset(nodeIndex: Int): Int = nodeIndex * nodeStride

  /**
   * Update the stats for a given (node, feature, bin) for ordered features, using the given label.
   * This uses a pre-computed node offset from [[getNodeOffset]].
   */
  def nodeUpdate(nodeOffset: Int, featureIndex: Int, binIndex: Int, label: Double): Unit = {
    val i = nodeOffset + featureOffsets(featureIndex) + binIndex * statsSize
    impurityAggregator.update(allStats, i, label)
  }

  /**
   * Pre-compute (node, feature) offset for use with [[nodeFeatureUpdate]].
   * For ordered features only.
   */
  def getNodeFeatureOffset(nodeIndex: Int, featureIndex: Int): Int = {
    require(!isUnordered(featureIndex),
      s"DTStatsAggregator.getNodeFeatureOffset is for ordered features only, but was called" +
      s" for unordered feature $featureIndex.")
    nodeIndex * nodeStride + featureOffsets(featureIndex)
  }

  /**
   * Pre-compute (node, feature) offset for use with [[nodeFeatureUpdate]].
   * For unordered features only.
   */
  def getLeftRightNodeFeatureOffsets(nodeIndex: Int, featureIndex: Int): (Int, Int) = {
    require(isUnordered(featureIndex),
      s"DTStatsAggregator.getLeftRightNodeFeatureOffsets is for unordered features only," +
        s" but was called for ordered feature $featureIndex.")
    val baseOffset = nodeIndex * nodeStride + featureOffsets(featureIndex)
    (baseOffset, baseOffset + numBins(featureIndex) * statsSize)
  }

  /**
   * Update the stats for a given (node, feature, bin), using the given label.
   * @param nodeFeatureOffset  For ordered features, this is a pre-computed (node, feature) offset
   *                           from [[getNodeFeatureOffset]].
   *                           For unordered features, this is a pre-computed
   *                           (node, feature, left/right child) offset from
   *                           [[getLeftRightNodeFeatureOffsets]].
   */
  def nodeFeatureUpdate(nodeFeatureOffset: Int, binIndex: Int, label: Double): Unit = {
    impurityAggregator.update(allStats, nodeFeatureOffset + binIndex * statsSize, label)
  }

  /**
   * For a given (node, feature), merge the stats for two bins.
   * @param nodeFeatureOffset  For ordered features, this is a pre-computed (node, feature) offset
   *                           from [[getNodeFeatureOffset]].
   *                           For unordered features, this is a pre-computed
   *                           (node, feature, left/right child) offset from
   *                           [[getLeftRightNodeFeatureOffsets]].
   * @param binIndex  The other bin is merged into this bin.
   * @param otherBinIndex  This bin is not modified.
   */
  def mergeForNodeFeature(nodeFeatureOffset: Int, binIndex: Int, otherBinIndex: Int): Unit = {
    impurityAggregator.merge(allStats, nodeFeatureOffset + binIndex * statsSize,
      nodeFeatureOffset + otherBinIndex * statsSize)
  }

  /**
   * Merge this aggregator with another, and returns this aggregator.
   * This method modifies this aggregator in-place.
   */
  def merge(other: DTStatsAggregator): DTStatsAggregator = {
    require(allStatsSize == other.allStatsSize,
      s"DTStatsAggregator.merge requires that both aggregators have the same length stats vectors."
      + s" This aggregator is of length $allStatsSize, but the other is ${other.allStatsSize}.")
    var i = 0
    // TODO: Test BLAS.axpy
    while (i < allStatsSize) {
      allStats(i) += other.allStats(i)
      i += 1
    }
    this
  }

}

private[tree] object DTStatsAggregator extends Serializable {

  /**
   * Combines two aggregates (modifying the first) and returns the combination.
   */
  def binCombOp(
      agg1: DTStatsAggregator,
      agg2: DTStatsAggregator): DTStatsAggregator = {
    agg1.merge(agg2)
  }

}
