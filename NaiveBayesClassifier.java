import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class NaiveBayesClassifier {
    private Integer spamCount;
    private Integer nonSpamCount;
    private Map<Integer, Integer> spamFeatureCount;
    private Map<Integer, Integer> nonSpamFeatureCount;
    private double spamPrior;
    private double nonSpamPrior;
    private Map<Integer, Double> conditionalSpam;
    private Map<Integer, Double> conditionalNotSpam;


    NaiveBayesClassifier(List<String[]> data, int labelIndex) throws IOException {
        spamCount = 0;
        nonSpamCount = 0;
        spamFeatureCount = new HashMap<>();
        nonSpamFeatureCount = new HashMap<>();
        conditionalSpam = new HashMap<>();
        conditionalNotSpam = new HashMap<>();

        for(int i = 0; i < 48; i++) {
            spamFeatureCount.put(i, 0);
            nonSpamFeatureCount.put(i, 0);
        }

        for(String[] values : data) {
            Integer label = Integer.valueOf(values[labelIndex]);
            if(label == 1) {
                spamCount++;
            } else {
                nonSpamCount++;
            }

            for(int i = 0; i < 48; i++) {
                if(Double.valueOf(values[i]) > 0) {
                    if(label == 1) spamFeatureCount.put(i, spamFeatureCount.get(i) + 1);
                    else nonSpamFeatureCount.put(i, nonSpamFeatureCount.get(i) + 1);
                }
            }
        }

        for(int i = 0; i < 48; i++) {
            conditionalSpam.put(i, (double) spamFeatureCount.get(i) / (double) spamCount);
            conditionalNotSpam.put(i, (double) nonSpamFeatureCount.get(i) / (double) nonSpamCount);
        }

        spamPrior = (double) spamCount/ ((double) spamCount + (double) nonSpamCount);
        nonSpamPrior = 1 - spamPrior;
    }

    public Integer predictClass(String[] values) {
        double spamProb = spamPrior;
        double nonSpamProb = nonSpamPrior;

        for(int i = 0; i < 48; i++) {
            if(Double.valueOf(values[i]) > 0) {
                spamProb *= conditionalSpam.get(i);
                nonSpamProb *= conditionalNotSpam.get(i);
            }
        }

        if(spamProb >= nonSpamProb) {
            return 1;
        } else {
            return 0;
        }
    }
}
