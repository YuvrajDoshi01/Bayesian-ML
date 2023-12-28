import java.util.*;

public class EMAlgorithm {

    public static void main(String[] args) {
        int numDataPoints = 300; // Adjust the number of data points as needed
        int numFeatures = 2; // You can change the number of features
        int numClusters = 3; // Adjust the number of clusters
        int numIterations = 5000;

        List<double[]> syntheticData = generateSyntheticData(numDataPoints, numFeatures, numClusters);

        Random random = new Random();

        List<MultivariateGaussianPDF> clusters = new ArrayList<>();
        double[] priorLikelihood = new double[numClusters];

        for(int i = 0; i < numClusters; i++) {
            double[] meanVectors = new double[numFeatures];
            for(int j = 0; j < numFeatures; j++) meanVectors[j] = random.nextDouble(10);

            double[][] covarianceMatrix = new double[numFeatures][numFeatures];
            for(int j = 0; j < numFeatures; j++) {
                for(int k = 0; k < numFeatures; k++) {
                    covarianceMatrix[j][k] = (j == k) ? 1 : 0;
                }
            }

            clusters.add(new MultivariateGaussianPDF(meanVectors, covarianceMatrix));
            priorLikelihood[i] = 1/ (double) numClusters;
        }

        double[][] clusterWeight = new double[numDataPoints][numClusters];

        for(int it = 0; it < numIterations; it++) {

            //compute new cluster weights for every data point
            for(int n = 0; n < numDataPoints; n++) {
                for(int k = 0; k < numClusters; k++) {
                    clusterWeight[n][k] = (priorLikelihood[k]) * clusters.get(k).computePDF(syntheticData.get(n));
                }

                // normalize fractional assignments
                double normalizationFactor = 0.0;
                for(int k = 0; k < numClusters; k++) normalizationFactor += clusterWeight[n][k];
                for(int k = 0; k < numClusters; k++) clusterWeight[n][k] /= normalizationFactor;
            }

            // re-estimation

            for(int k = 0; k < numClusters; k++) {
                double sumOfClusterWeight = 0.0;
                double[] sumOfDataPointsWeighedByClusterWeight = new double[numFeatures];
                double[][] sumOuterProduct = new double[numFeatures][numFeatures];

                for(int n = 0; n < numDataPoints; n++) {
                    sumOfClusterWeight += clusterWeight[n][k];

                    for(int j = 0; j < numFeatures; j++) {
                        sumOfDataPointsWeighedByClusterWeight[j] += clusterWeight[n][k] * syntheticData.get(n)[j];
                    }

                    double[] deviation = new double[numFeatures];
                    for (int i = 0; i < numFeatures; i++) {
                        deviation[i] = syntheticData.get(n)[i] - clusters.get(k).getMeanVector()[i];
                    }

                    for (int i = 0; i < numFeatures; i++) {
                        for (int j = 0; j < numFeatures; j++) {
                            sumOuterProduct[i][j] += clusterWeight[n][k] * deviation[i] * deviation[j];
                        }
                    }

                }

                priorLikelihood[k] = sumOfClusterWeight/numDataPoints;

                double[] newMeans = new double[numFeatures];
                for(int j = 0; j < numFeatures; j++) {
                    newMeans[j] = sumOfDataPointsWeighedByClusterWeight[j]/sumOfClusterWeight;
                }
                clusters.get(k).setMeanVector(newMeans);

                for(int i = 0; i < numFeatures; i++) {
                    for(int j = 0; j < numFeatures; j++) {
                        sumOuterProduct[i][j] /= sumOfClusterWeight;
                    }
                }
                clusters.get(k).setCovarianceMatrix(sumOuterProduct);
            }

        }

        // print cluster values
        int p = 0;
        for(MultivariateGaussianPDF c : clusters) {
            p++;
            System.out.println("\n\nPrior Likelihood for cluster " + p + " is " + priorLikelihood[p - 1]);
            System.out.println("The mean vector for cluster " + p);
            double[] meanVector = c.getMeanVector();
            for(double d : meanVector) {
                System.out.print(d + " ");
            }

            System.out.println("\n\nThe covariance matrix for cluster " + p);
            double[][] covMat = c.getCovarianceMatrix();
            for(int i = 0; i < numFeatures; i++) {
                for(int j = 0; j < numFeatures; j++) {
                    System.out.print(covMat[i][j] + " ");
                }
                System.out.print("\n");
            }
        }
    }

    public static List<double[]> generateSyntheticData(int numDataPoints, int numFeatures, int numClusters) {
        List<double[]> data = new ArrayList<>();
        Random random = new Random();
        Integer[] numDataPointsInCluster = {0, 0, 0};

        for (int i = 0; i < numDataPoints; i++) {
            int cluster = random.nextInt(numClusters);
            double[] point = new double[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                // Generate data points for each feature based on the cluster
                if (cluster == 0) {
                    point[j] = random.nextGaussian() + 1.0;
                    numDataPointsInCluster[0]++;
                } else if (cluster == 1) {
                    point[j] = random.nextGaussian() + 4.0;
                    numDataPointsInCluster[1]++;
                } else {
                    point[j] = random.nextGaussian() + 9.0;
                    numDataPointsInCluster[2]++;
                }
            }
            data.add(point);
        }
        System.out.println("Distribution of points in different distributions : ");
        for(int i = 0 ; i < numClusters; i++) {
            System.out.println("Cluster " + (i + 1) + " has " + (double) numDataPointsInCluster[i]/ (2.0 * (double) numDataPoints) + " of the total points");
        }
        return data;
    }
}
