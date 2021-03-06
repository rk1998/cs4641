import opt.OptimizationAlgorithm;
import shared.ConvergenceTrainer;
import shared.DataSet;
import shared.Instance;
import shared.SumOfSquaresError;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import shared.filt.RandomOrderFilter;
import shared.filt.TestTrainSplitFilter;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class WineNNBackProp {
    private static DecimalFormat df = new DecimalFormat("0.000");
    public static void main(String[] args) {
        BackPropagationNetworkFactory factory =
                new BackPropagationNetworkFactory();
        List<Double> trainingErrorResults = new ArrayList<>();
        List<Double> testingErrorResults = new ArrayList<>();
        List<Double> trainingTimeResults = new ArrayList<>();
        List<Double> testingTimeResults = new ArrayList<>();


        double[] dataSetSizes = {0.1, 0.3, 0.5, 0.7, 0.9, 1.0};
//        BackPropagationNetwork network = factory.createClassificationNetwork(
//                new int[] { 11, 6, 4, 3, 2, 11 });
        DataSet set = new DataSet(readWineData());
        new RandomOrderFilter().filter(set);
        TestTrainSplitFilter filter = new TestTrainSplitFilter(70);
        filter.filter(set);
        DataSet training = filter.getTrainingSet();
        DataSet testing = filter.getTestingSet();
//        ConvergenceTrainer trainer = new ConvergenceTrainer(
//                new BatchBackPropagationTrainer(set, network,
//                        new SumOfSquaresError(), new RPROPUpdateRule()));
        for(double percent : dataSetSizes) {
            double correct = 0;
            double incorrect = 0;
            int endIndex = (int) (training.getInstances().length * percent);
            BackPropagationNetwork network = factory.createClassificationNetwork(
                    new int[] { 11, 6, 4, 3, 2, 11 });
            ConvergenceTrainer trainer = new ConvergenceTrainer(
                    new BatchBackPropagationTrainer(set, network,
                            new SumOfSquaresError(), new RPROPUpdateRule()));
            Instance[] trainingInstance = Arrays.copyOfRange(
                    training.getInstances(), 0, endIndex);
            DataSet subSample = new DataSet(trainingInstance);
            double start = System.nanoTime(), end, trainingTime, testingTime = 0;
            double trainingError = train(trainer, network, "Backpropagation", 2000, subSample);
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);
            Instance[] testInstances = testing.getInstances();
            start = System.nanoTime();
            for(int i = 0; i < testInstances.length; i++) {
                network.setInputValues(testInstances[i].getData());
                network.run();
//                predicted = Double.parseDouble(network.getOutputValues().toString());
//                actual = Double.parseDouble(testInstances[i].getLabel().toString());
//                predicted = getOutputFromString(network.getOutputValues().toString());
//                actual = getOutputFromString(testInstances[i].getLabel().toString());
                Instance example = new Instance(network.getOutputValues());
                Instance output = testInstances[i].getLabel();
//                double temp = Math.abs(actual - predicted) < 0.5 ? correct++ : incorrect++;
                double temp = example.getData().argMax() == output.getData().argMax()
                        ? correct++: incorrect++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10, 9);
            double percentageCorrect = (correct / (incorrect + correct));
            trainingErrorResults.add(trainingError);
            testingErrorResults.add(1.0 - percentageCorrect);
            trainingTimeResults.add(trainingTime);
            testingTimeResults.add(testingTime);
            System.out.println("\n\nData Set Size: " + df.format(percent * training.getInstances().length));
        }
        reportResults(trainingErrorResults, testingErrorResults,
                trainingTimeResults, testingTimeResults);

    }

    private static double train(ConvergenceTrainer trainer,
                                BackPropagationNetwork network,
                                String algoName,
                                int iterations, DataSet training) {
        Instance[] trainingInstances = training.getInstances();
        //System.out.println("\nTraining with " + algoName + "\n\n");
        double error = 0;
        for (int i = 0; i < iterations; i++) {
            int incorrect = 0;
            trainer.train();
            //System.out.println("Convergence in " + trainer.getIterations() + "iterations");
            double predicted = 0.0;
            double trainError = 0.0;
            double actual = 0.0;
            for (int j = 0; j < trainingInstances.length; j++) {
                network.setInputValues(trainingInstances[j].getData());
                network.run();

                Instance output = trainingInstances[j].getLabel();
                Instance example = new Instance(network.getOutputValues());
//                example.setLabel(new Instance(
//                        Double.parseDouble(
//                                network.getOutputValues().toString())));
                //trainError += errorMetric.value(output, example);
//                predicted = Double.parseDouble(output.toString());
//                actual = Double.parseDouble(example.getLabel().toString());
//                predicted = getOutputFromString(output.toString());
//                actual = getOutputFromString(network.getOutputValues().toString());
                incorrect += output.getData().argMax() == example.getData().argMax() ? 0 : 1;
            }
//            if(i % 100 == 0) {
//                System.out.println("\n\nTrain Error: "
//                        + df.format(incorrect / (double) trainingInstances.length) + "\n");
//            }
            error += (incorrect / (double) trainingInstances.length);
            //System.out.println("\n\nTest Error: " + df.format(testError/ (double) 1470) + "\n");

        }
        double averageError = error / (double) iterations;
        return averageError;
    }

    private static void reportResults(List<Double> trainingError,
                                      List<Double> testingError,
                                      List<Double> trainingTime,
                                      List<Double> testingTime) {
        System.out.println("\n\nTraining Error");
        for(double trainError: trainingError) {
            System.out.println(trainError);
        }
        System.out.println("\n\nTesting Error");
        for(double testError: testingError) {
            System.out.println(testError);
        }
        System.out.println("\n\nTraining Times");
        for(double time: trainingTime) {
            System.out.println(time);
        }
        System.out.println("\n\nTesting Times");
        for(double time: testingTime) {
            System.out.println(time);

        }


    }

    /**
     * Reads in winequality_white_scaled.csv and returns an array of Instance objects
     * @return array of Instance objects
     */
    private static Instance[] readWineData() {
        double[][][] attributes = new double[4898][][];
        String line = "";
        try {
            BufferedReader br = new BufferedReader(new FileReader(
                    "src/data/winequality_white_scaled.csv"));
            int i = 0;
            br.readLine(); // skip header line
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                attributes[i] = new double[2][];
                attributes[i][0] = new double[11];
                attributes[i][1] = new double[1];

                for (int j = 0; j < 11; j++) {
                    attributes[i][0][j] = Double.parseDouble(values[j]);
                }
                attributes[i][1][0] = Double.parseDouble(values[11]);
                i++;

            }
            br.close();
        } catch (FileNotFoundException fe) {
            System.out.println("file not found");

        } catch (IOException e) {
            System.out.println("Error reading file");
        }
        Instance[] instances = new Instance[attributes.length];
        for (int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            int c = (int) attributes[i][1][0];
            double[] classes = new double[11];
            classes[c] = 1.0;
            //instances[i].setLabel(new Instance(attributes[i][1][0]));
            instances[i].setLabel(new Instance(classes));
        }
        return instances;

    }
}
