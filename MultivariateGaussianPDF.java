public class MultivariateGaussianPDF {
    private double[] meanVector;
    private double[][] covarianceMatrix;

    public MultivariateGaussianPDF(double[] meanVector, double[][] covarianceMatrix) {
        this.covarianceMatrix = covarianceMatrix;
        this.meanVector = meanVector;
    }

    public double[] getMeanVector() {
        return meanVector;
    }

    public double[][] getCovarianceMatrix() {
        return covarianceMatrix;
    }

    public void setMeanVector(double[] meanVector){
        this.meanVector = meanVector;
    }

    public void setCovarianceMatrix(double[][] covarianceMatrix) {
        this.covarianceMatrix = covarianceMatrix;
    }

    private double[][] findInverse(double[][] matrix) {
        int n = matrix.length;

        // Check if the matrix is square
        if (n != matrix[0].length) {
            throw new IllegalArgumentException("Matrix must be square.");
        }

        if(n == 1) {
            return new double[][]{{1 / matrix[0][0]}};
        }

        // Create an augmented matrix [matrix | I]
        double[][] augmentedMatrix = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmentedMatrix[i][j] = matrix[i][j];
                augmentedMatrix[i][j + n] = (i == j) ? 1.0 : 0.0;
            }
        }

        // Apply Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Scale the pivot row
            double pivot = augmentedMatrix[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmentedMatrix[i][j] /= pivot;
            }

            // Eliminate other rows
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmentedMatrix[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmentedMatrix[k][j] -= factor * augmentedMatrix[i][j];
                    }
                }
            }
        }

        // Extract the inverse matrix [I | inverse]
        double[][] inverseMatrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverseMatrix[i][j] = augmentedMatrix[i][j + n];
            }
        }

        return inverseMatrix;
    }

    private double calculateSemidefinite(double[][] matrix, double[] vector) {
        int n = matrix.length;
        int m = vector.length;

        if (n != m) {
            throw new IllegalArgumentException("Matrix and vector dimensions must match.");
        }

        double result = 0;

        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                result += matrix[i][j] * vector[i] * vector[j];
            }
        }

        return result;
    }

    private double calculateDeterminant(double[][] matrix) {
        int n = matrix.length;

        if (n == 1) {
            return matrix[0][0];
        }

        if (n != matrix[0].length) {
            throw new IllegalArgumentException("Matrix must be square.");
        }

        if (n == 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }

        double determinant = 0;

        for (int i = 0; i < n; i++) {
            double[][] subMatrix = new double[n - 1][n - 1];

            for (int j = 1; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if (k < i) {
                        subMatrix[j - 1][k] = matrix[j][k];
                    } else if (k > i) {
                        subMatrix[j - 1][k - 1] = matrix[j][k];
                    }
                }
            }

            determinant += matrix[0][i] * Math.pow(-1, i) * calculateDeterminant(subMatrix);
        }

        return determinant;
    }

    public double computePDF(double[] dataVector) {
        int dimension = meanVector.length;

        if (dataVector.length != dimension) {
            throw new IllegalArgumentException("Data vector dimensions do not match the mean vector dimensions.");
        }

        // Calculate the determinant of the covariance matrix
        double covarianceDeterminant = calculateDeterminant(covarianceMatrix);

        // Check if the covariance matrix is invertible
        if (covarianceDeterminant == 0.0) {
            throw new IllegalArgumentException("Covariance matrix is singular (non-invertible).");
        }

        // Calculate the Mahalanobis distance
        double[] diff = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            diff[i] = dataVector[i] - meanVector[i];
        }

        // Calculate the PDF
        double exponent = -0.5 * calculateSemidefinite(findInverse(covarianceMatrix), diff);
        double prefactor = 1.0 / (Math.sqrt(Math.pow(2 * Math.PI, dimension) * covarianceDeterminant));

        return prefactor * Math.exp(exponent);
    }
}
