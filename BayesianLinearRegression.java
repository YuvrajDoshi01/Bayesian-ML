import java.util.*;

public class BayesianLinearRegression {

    public static void main(String[] args) {
        int numDataPoints = 100;
        List<double[]> data = randomRegressionGenerator(numDataPoints);
        double[] results = calculateLinearRegression(data);


        double[] residuals = new double[numDataPoints];
        for(int i = 0; i < numDataPoints; i++) {
            residuals[i] = -(results[0] * data.get(0)[i] + results[1]) + data.get(1)[i]; // yhat - y;
        }

        List<List<PriorDistribution>> allPriors = calculateAllPriors(results, residuals);

        List<Combination> jointDistributions = generateJointDistributions(allPriors);

        jointDistributions = generateLikelihoods(jointDistributions, data);

        // Bayesian Update
        double posteriorSum = 0.0;
        for(Combination c : jointDistributions) {
            c.posterior = c.priorProbability * c.likelihood;
            //System.out.println(c.priorProbability + ", " + c.likelihood);
            posteriorSum += c.posterior;
        }

        for(Combination c : jointDistributions) {
            c.posterior /= posteriorSum;
        }

        double[] resultSlope = retrieveMeanAndStd(allPriors.get(0), jointDistributions);
        double[] resultIntercept = retrieveMeanAndStd(allPriors.get(1), jointDistributions);
        double[] resultSigma = retrieveMeanAndStd(allPriors.get(2), jointDistributions);



    }

    public static double[] retrieveMeanAndStd(List<PriorDistribution> priors, List<Combination> combinations) {
        double[] result = new double[2];

        Map<Double, Double> marginalMap = new HashMap<>();
        double[] marginal = new double[priors.size()];

        if(priors.get(0).name.equals("slope")) {
            for(Combination c : combinations) {
                if(marginalMap.containsKey(c.slope)) {
                    marginalMap.put(c.slope, marginalMap.get(c.slope) + c.posterior);
                } else {
                    marginalMap.put(c.slope, c.posterior);
                }
            }
        } else if(priors.get(0).name.equals("intercept")) {
            for(Combination c : combinations) {
                if(marginalMap.containsKey(c.intercept)) {
                    marginalMap.put(c.intercept, marginalMap.get(c.intercept) + c.posterior);
                } else {
                    marginalMap.put(c.intercept, c.posterior);
                }
            }
        } else if(priors.get(0).name.equals("sigma")) {
            for(Combination c : combinations) {
                if(marginalMap.containsKey(c.sigma)) {
                    marginalMap.put(c.sigma, marginalMap.get(c.sigma) + c.posterior);
                } else {
                    marginalMap.put(c.sigma, c.posterior);
                }
            }
        }
        int i = 0;
        result[0] = 0;
        double maxMarginal = 0;
        for(double d : marginalMap.keySet()) {
            marginal[i] = marginalMap.get(d);
            //System.out.println(d +  " " + marginalMap.get(d));
            if(marginal[i] > maxMarginal) result[0] = d;
            //System.out.println(result[0]);
            i++;

        }
        result[1] = calculateStandardDeviation(marginal);

        System.out.println("\nThe most likely estimate for " + priors.get(0).name + " is : " +  result[0]);

        return result;
    }

    public static List<List<PriorDistribution>> calculateAllPriors(double[] results, double[] residuals) {
        // Create a range of values for the slope
        double slopeMin = results[0] * 0.8;
        double slopeMax = results[0] * 1.2;
        int numSlopeValues = 60;
        List<Double> dataSlope = createRange(slopeMin, slopeMax, numSlopeValues);
        List<PriorDistribution> priorSlope = makeUninformativePrior("slope", dataSlope);

        // Create a range of values for the intercept
        double interceptMin = results[1] * 0.8;
        double interceptMax = results[1]* 1.2;
        int numInterceptValues = 60;
        List<Double> dataIntercept = createRange(interceptMin, interceptMax, numInterceptValues);
        List<PriorDistribution> priorIntercept = makeUninformativePrior("intercept", dataIntercept);

        // Create a range of values for the sigma
        double stddev = calculateStandardDeviation(residuals);

        double sigmaMin = stddev * 0.8;
        double sigmaMax = stddev * 1.2;
        int numSigmaValues = 60;
        List<Double> dataSigma = createRange(sigmaMin, sigmaMax, numSigmaValues);
        List<PriorDistribution> priorSigma = makeUninformativePrior("sigma", dataSigma);

        // Now 'priorSlope', 'priorIntercept', and 'priorSigma' contain uninformative priors
        return new ArrayList<> (Arrays.asList(priorSlope, priorIntercept, priorSigma));
    }

    // generate all possible joint distributions
    public static List<Combination> generateJointDistributions(List<List<PriorDistribution>> allPriors) {

        List<Combination> combinations = new ArrayList<>();

        for (PriorDistribution slope : allPriors.get(0)) {
            double probSlope = getPriorProbability(allPriors.get(0), "slope", slope.value);

            for (PriorDistribution intercept : allPriors.get(1)) {
                double probIntercept = getPriorProbability(allPriors.get(1), "intercept", intercept.value);

                for (PriorDistribution sigma : allPriors.get(2)) {
                    double probSigma = getPriorProbability(allPriors.get(2), "sigma", sigma.value);

                    double prob = probSlope * probIntercept * probSigma;

                    Combination combination = new Combination(slope.value, intercept.value, sigma.value, prob);
                    //System.out.println(combination.slope + " " +  combination.intercept + " " + combination.sigma + " " + combination.priorProbability);
                    combinations.add(combination);
                }
            }
        }

        return combinations;
    }

    public static List<Combination> generateLikelihoods(List<Combination> combinations, List<double[]> data) {
        for(Combination c : combinations) {
            double[] predictions = new double[data.size()];
            double[] residuals = new double[data.size()];

            for (int i = 0; i < data.size(); i++) {
                double prediction = c.slope * data.get(0)[i] + c.intercept;
                predictions[i] = prediction;
                double residual = data.get(1)[i] - prediction;
                residuals[i] = residual;
            }

            MultivariateGaussianPDF normalDistribution = new MultivariateGaussianPDF(new double[]{0}, new double[][]{{c.sigma}});
            double likelihood = 1.0;
            for (int i = 0; i < data.size(); i++) {
                double likelihoodTerm = normalDistribution.computePDF(new double[]{residuals[i]});
                //System.out.println(likelihoodTerm);
                likelihood *= likelihoodTerm;
            }

            c.likelihood = likelihood;
        }

        return combinations;
    }
    private static double getPriorProbability(List<PriorDistribution> priorList, String name, double value) {
        for (PriorDistribution prior : priorList) {
            if (prior.name.equals(name) && prior.value == value) {
                return prior.probability;
            }
        }
        return 0.0;
    }

    // combinations class
    private static class Combination {
        double slope;
        double intercept;
        double sigma;
        double priorProbability;

        double likelihood = 1.0;
        double posterior = 0.0;

        Combination(double slope, double intercept, double sigma, double priorProbability) {
            this.slope = slope;
            this.intercept = intercept;
            this.sigma = sigma;
            this.priorProbability = priorProbability;
        }
    }

    // Function to create a range of values
    private static List<Double> createRange(double min, double max, int numValues) {
        List<Double> data = new ArrayList<>();
        double step = (max - min) / (numValues - 1);
        for (int i = 0; i < numValues; i++) {
            data.add(min + i * step);
        }
        return data;
    }

    // Function to create uninformative priors
    private static List<PriorDistribution> makeUninformativePrior(String name, List<Double> data) {
        List<PriorDistribution> priorDistributions = new ArrayList<>();
        double probability = 1.0 / data.size();
        for (double value : data) {
            priorDistributions.add(new PriorDistribution(name, value, probability));
        }
        return priorDistributions;
    }

    // Define a class to represent a prior distribution
    private static class PriorDistribution {
        String name;
        double value;
        double probability;

        PriorDistribution(String name, double value, double probability) {
            this.name = name;
            this.value = value;
            this.probability = probability;
        }
    }

    public static double calculateStandardDeviation(double[] data) {
        if (data.length == 0) {
            throw new IllegalArgumentException("Input array is empty.");
        }

        // Step 1: Calculate the mean
        double sum = 0.0;
        for (double value : data) {
            sum += value;
        }
        double mean = sum / data.length;

        // Step 2: Calculate the sum of squared differences
        double sumOfSquaredDifferences = 0.0;
        for (double value : data) {
            double difference = value - mean;
            sumOfSquaredDifferences += difference * difference;
        }

        // Step 3: Divide by the number of elements
        double variance = sumOfSquaredDifferences / data.length;

        // Step 4: Take the square root to get the standard deviation
        double stddev = Math.sqrt(variance);

        return stddev;
    }

    public static double[] calculateLinearRegression(List<double[]> data) {
        int n = data.get(0).length;
        double sumX = 0;
        double sumY = 0;
        double sumXY = 0;
        double sumX2 = 0;

        for (int i = 0; i < n; i++) {
            double xi = data.get(0)[i];
            double yi = data.get(1)[i];
            sumX += xi;
            sumY += yi;
            sumXY += xi * yi;
            sumX2 += xi * xi;
        }

        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double intercept = (sumY - (slope * sumX)) / n;

        double[] coefficients = { slope, intercept };
        System.out.println(slope + " " + intercept);
        return coefficients;
    }

    public static List<double[]> randomRegressionGenerator(int numDataPoints) {
        Random rand = new Random();
        List<double[]> regression = new ArrayList<>();
        regression.add(new double[numDataPoints]);
        regression.add(new double[numDataPoints]);
        for(int i = 0; i < numDataPoints; i++) {
            regression.get(0)[i] = rand.nextDouble(-200, 200);
        }

        double slope = rand.nextDouble(0.5, 2);
        double intercept = rand.nextDouble();

        for(int i = 0; i < numDataPoints; i++) {
            regression.get(1)[i] = slope * regression.get(0)[i] + intercept + rand.nextGaussian();
        }

        System.out.println("slope : " + slope);
        System.out.println("intercept : " + intercept);
        return regression;
    }
}
