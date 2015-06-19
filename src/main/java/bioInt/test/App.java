package bioInt.test;

import java.io.IOException;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.core.io.ClassPathResource;

import com.amd.aparapi.Kernel;

/**
 * Hello world!
 */
public class App
{
	public static void main(String[] args) throws IOException, InterruptedException
	{
		final DataSetIterator trainIterator = getDataSetIterator("train.csv");
		DataSetIterator validationIterator = getDataSetIterator("validate.csv");

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().layer(new RBM()).nIn(trainIterator.inputColumns())
				.nOut(trainIterator.totalOutcomes()).visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED)
				.weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1)).constrainGradientToUnitNorm(true).iterations(5)
				.batchSize(10).lossFunction(LossFunctions.LossFunction.RMSE_XENT).learningRate(1e-1f).momentum(0.9)
				// .momentumAfter(Collections.singletonMap(3, 0.9))
				// .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
				.list(2).hiddenLayerSizes(100).useDropConnect(true).override(1, new ClassifierOverride()).build();

		final MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		Kernel kernel = new Kernel()
		{
			@Override
			public void run()
			{
				model.fit(trainIterator);
			}
		};

		kernel.execute(8);

		Evaluation eval = new Evaluation();

		while (validationIterator.hasNext())
		{
			DataSet validation = validationIterator.next();
			INDArray predict = model.output(validation.getFeatureMatrix());
			eval.eval(validation.getLabels(), predict);
		}

		System.out.printf("Score: %s\n", eval.stats());
	}

	static DataSetIterator getDataSetIterator(String resource) throws IOException, InterruptedException
	{
		RecordReader recordReader = new CSVRecordReader();
		FileSplit csv = new FileSplit(new ClassPathResource(resource).getFile());
		recordReader.initialize(csv);
		return new RecordReaderDataSetIterator(recordReader, 10, 0, 2);
	}
}
