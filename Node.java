import java.util.*;

class Node {
    String name;
    List<Node> parents;
    List<Double> prob;

    Node(String name, List<Node> parents) {
        this.name = name;
        this.parents = parents;
        this.prob = new ArrayList<>();
    }
}