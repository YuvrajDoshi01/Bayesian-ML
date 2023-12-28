import java.util.*;

public class BayesianBeliefNetworks {
    public static double probability(Node query, int queryValue) {
        int index = 0;
        double resultProb = 1.0;

        Scanner in = new Scanner(System.in);

        int i = 0;
        for(Node par : query.parents) {
            System.out.println("Enter truth value of evidence node " + par.name + " (1/0) :- ");
            int truth = in.nextInt();
            index += Math.pow(2, i) * truth;
            resultProb *= probability(par, truth);
            i++;
        }

        resultProb *= (queryValue == 1) ? query.prob.get(index) : (1 - query.prob.get(index));

        return resultProb;
    }

    public static void main(String[] args) {
        Node nodeA = new Node("A", new ArrayList<>());
        Node nodeB = new Node("B", new ArrayList<>());
        Node nodeC = new Node("C", Arrays.asList(nodeA, nodeB));

        // conditional probability table

        nodeA.prob.add(0.8);
        nodeB.prob.add(0.9);

        nodeC.prob.add(0.13);
        nodeC.prob.add(0.17);
        nodeC.prob.add(0.23);
        nodeC.prob.add(0.31);

        System.out.println("Finding probability of hypothesis given evidence");
        Node query = nodeC;
        Scanner in = new Scanner(System.in);

        System.out.println("Enter Truth value of hypothesis node" + query.name +" (1/0) ");
        int queryValue = in.nextInt();
        double p = probability(query, queryValue);
        System.out.println(p);
    }
}