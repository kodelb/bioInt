package bioInt.test;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.Collections;

/**
 * Hello world!
 */
public class App {
    public static void main(String[] args) throws IOException, InterruptedException {
        RecordReader recordReader = new CSVRecordReader();
        FileSplit csv = new FileSplit(new ClassPathResource("train.csv").getFile());
        recordReader.initialize(csv);
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 10, 0, 2);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(iterator.inputColumns())
                .nOut(iterator.totalOutcomes())
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new UniformDistribution(0, 1))
                .constrainGradientToUnitNorm(true)
                .iterations(5)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f)
                .momentum(0.9)
                //.momentumAfter(Collections.singletonMap(3, 0.9))
                //.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(2)
                .hiddenLayerSizes(400)
                .override(1, new ClassifierOverride())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.fit(iterator);

    }
}
