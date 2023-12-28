import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NaiveBayesDriver {
    public static void main(String[] args) throws IOException {
        List<String[]> data = new ArrayList<>();

        String fileName = "C:\\Users\\Ajay kumar sinha\\IdeaProjects\\MachineLearningAssignmentOne\\src\\spambase.csv";
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        String line;
        while((line = br.readLine()) != null) {
            String[] values = line.split(",");
            data.add(values);
        }
        Collections.shuffle(data); // randomise input data

        int n = (int) (0.7 * (double) data.size());
        List<String[]> trainData = data.subList(0, n);
        List<String[]> testData = data.subList(n + 1, data.size() - 1);

        NaiveBayesClassifier nbc = new NaiveBayesClassifier(trainData, trainData.get(0).length - 1);

        int correctPredictions = 0;

        for(String[] row : testData) {
            if(Integer.valueOf(row[row.length - 1]) == nbc.predictClass(row)) {
                correctPredictions++;
            }
        }

        double result = (double) correctPredictions/ (double) testData.size();

        System.out.println("The prediction rate of the Naive Bayes Classifier is :- " + result);
    }
}
