package bioInt.test;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws IOException, InterruptedException {
        RecordReader recordReader = new CSVRecordReader();
        FileSplit csv = new FileSplit(new ClassPathResource("train.csv").getFile());
        recordReader.initialize(csv);
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 5, 0, 2);
        DataSet next = iterator.next();
        System.out.println(next.numInputs());
        System.out.println(next.numExamples());
        System.out.println(next.numOutcomes());
        System.out.println(next);
    }
}
