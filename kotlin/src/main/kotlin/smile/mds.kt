/*******************************************************************************
 * Copyright (c) 2010-2020 Haifeng Li. All rights reserved.
 *
 * Smile is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * Smile is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Smile.  If not, see <https://www.gnu.org/licenses/>.
 ******************************************************************************/

package smile.mds

/**
 * Classical multidimensional scaling, also known as principal coordinates
 * analysis. Given a matrix of dissimilarities (e.g. pairwise distances), MDS
 * finds a set of points in low dimensional space that well-approximates the
 * dissimilarities in A. We are not restricted to using a Euclidean
 * distance metric. However, when Euclidean distances are used MDS is
 * equivalent to PCA.
 *
 * @param proximity the nonnegative proximity matrix of dissimilarities. The
 *                  diagonal should be zero and all other elements should be positive and
 *                  symmetric. For pairwise distances matrix, it should be just the plain
 *                  distance, not squared.
 * @param k the dimension of the projection.
 * @param positive if true, estimate an appropriate constant to be added
 *            to all the dissimilarities, apart from the self-dissimilarities, that
 *            makes the learning matrix positive semi-definite. The other formulation of
 *            the additive constant problem is as follows. If the proximity is
 *            measured in an interval scale, where there is no natural origin, then there
 *            is not a sympathy of the dissimilarities to the distances in the Euclidean
 *            space used to represent the objects. In this case, we can estimate a constant c
 *            such that proximity + c may be taken as ratio data, and also possibly
 *            to minimize the dimensionality of the Euclidean space required for
 *            representing the objects.
 */
fun mds(proximity: Array<DoubleArray>, k: Int, positive: Boolean = false): MDS {
    return MDS.of(proximity, k, positive)
}

/**
 * Kruskal's nonmetric MDS. In non-metric MDS, only the rank order of entries
 * in the proximity matrix (not the actual dissimilarities) is assumed to
 * contain the significant information. Hence, the distances of the final
 * configuration should as far as possible be in the same rank order as the
 * original data. Note that a perfect ordinal re-scaling of the data into
 * distances is usually not possible. The relationship is typically found
 * using isotonic regression.
 *
 * @param proximity the nonnegative proximity matrix of dissimilarities. The
 *                  diagonal should be zero and all other elements should be positive and symmetric.
 * @param k the dimension of the projection.
 * @param tol tolerance for stopping iterations.
 * @param maxIter maximum number of iterations.
 */
fun isomds(proximity: Array<DoubleArray>, k: Int, tol: Double = 0.0001, maxIter: Int = 200): IsotonicMDS {
    return IsotonicMDS.of(proximity, k, tol, maxIter)
}

/**
 * The Sammon's mapping is an iterative technique for making interpoint
 * distances in the low-dimensional projection as close as possible to the
 * interpoint distances in the high-dimensional object. Two points close
 * together in the high-dimensional space should appear close together in the
 * projection, while two points far apart in the high dimensional space should
 * appear far apart in the projection. The Sammon's mapping is a special case of
 * metric least-square multidimensional scaling.
 *
 * Ideally when we project from a high dimensional space to a low dimensional
 * space the image would be geometrically congruent to the original figure.
 * This is called an isometric projection. Unfortunately it is rarely possible
 * to isometrically project objects down into lower dimensional spaces. Instead of
 * trying to achieve equality between corresponding inter-point distances we
 * can minimize the difference between corresponding inter-point distances.
 * This is one goal of the Sammon's mapping algorithm. A second goal of the Sammon's
 * mapping algorithm is to preserve the topology as best as possible by giving
 * greater emphasize to smaller interpoint distances. The Sammon's mapping
 * algorithm has the advantage that whenever it is possible to isometrically
 * project an object into a lower dimensional space it will be isometrically
 * projected into the lower dimensional space. But whenever an object cannot
 * be projected down isometrically the Sammon's mapping projects it down to reduce
 * the distortion in interpoint distances and to limit the change in the
 * topology of the object.
 *
 * The projection cannot be solved in a closed form and may be found by an
 * iterative algorithm such as gradient descent suggested by Sammon. Kohonen
 * also provides a heuristic that is simple and works reasonably well.
 *
 * @param proximity the nonnegative proximity matrix of dissimilarities. The
 *                  diagonal should be zero and all other elements should be positive and symmetric.
 * @param k         the dimension of the projection.
 * @param lambda    initial value of the step size constant in diagonal Newton method.
 * @param tol       tolerance for stopping iterations.
 * @param stepTol   tolerance on step size.
 * @param maxIter   maximum number of iterations.
 */
fun sammon(proximity: Array<DoubleArray>, k: Int, lambda: Double = 0.2, tol: Double = 0.0001, stepTol: Double = 0.001, maxIter: Int = 100): SammonMapping {
    return SammonMapping.of(proximity, k, lambda, tol, stepTol, maxIter)
}
