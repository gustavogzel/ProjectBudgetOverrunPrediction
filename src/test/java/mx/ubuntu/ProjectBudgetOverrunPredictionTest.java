package mx.ubuntu;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class ProjectBudgetOverrunPredictionTest {

    private OnlineLogisticRegression model;

    @BeforeEach
    public void setUp() throws Exception {
        List<Project> projects = Arrays.asList(
            new Project("Construction", "North", 1000, 5000, false),
            new Project("IT", "South", 200, 1500, true),
            new Project("Consulting", "East", 150, 1000, false),
            new Project("Construction", "West", 1200, 6000, true),
            new Project("IT", "North", 250, 2000, false)
        );

        String csvFilePath = "projects_test.csv";
        saveToCSV(projects, csvFilePath);

        model = trainModel(projects);
        
    }

    @Test
    public void testModelNotNull() {
        assertNotNull(model);
    }

    @Test
    public void testPrediction() {
        Project newProject = new Project("Construction", "South", 900, 5500, false);
        boolean prediction = predictBudgetOverrun(model, newProject);
        System.out.println("Predicted Budget Overrun: " + prediction);
        // Assert the prediction to be within a reasonable range
        assertEquals(true, prediction);
    }

    private void saveToCSV(List<Project> projects, String filePath) throws IOException {
        try (FileWriter writer = new FileWriter(filePath)) {
            writer.write("projectType,region,projectSize,pastProjectCost,budgetOverrun\n");
            for (Project project : projects) {
                writer.write(String.format("%s,%s,%f,%f,%b\n",
                        project.projectType(),
                        project.region(),
                        project.projectSize(),
                        project.pastProjectCost(),
                        project.budgetOverrun()));
            }
        }
    }

    private OnlineLogisticRegression trainModel(List<Project> projects) {
        OnlineLogisticRegression model = new OnlineLogisticRegression(2, 4, new L1())
                .lambda(1e-4)
                .learningRate(1);

        for (Project project : projects) {
            Vector input = new DenseVector(new double[]{
                encodeProjectType(project.projectType()),
                encodeRegion(project.region()),
                project.projectSize(),
                project.pastProjectCost()
            });
            int label = project.budgetOverrun() ? 1 : 0;
            model.train(label, input);
        }
        return model;
    }

    private boolean predictBudgetOverrun(OnlineLogisticRegression model, Project project) {
        Vector input = new DenseVector(new double[]{
            encodeProjectType(project.projectType()),
            encodeRegion(project.region()),
            project.projectSize(),
            project.pastProjectCost()
        });
        double prediction = model.classifyScalar(input);
        return prediction > 0.5;
    }

    private double encodeProjectType(String projectType) {
        return switch (projectType) {
            case "Construction" -> 1.0;
            case "IT" -> 2.0;
            case "Consulting" -> 3.0;
            default -> 0.0;
        };
    }

    private double encodeRegion(String region) {
        return switch (region) {
            case "North" -> 1.0;
            case "South" -> 2.0;
            case "East" -> 3.0;
            case "West" -> 4.0;
            default -> 0.0;
        };
    }
}

