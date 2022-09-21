package experiment;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessing.text.KerasTokenizer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.resources.Resources;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.cpu.nativecpu.buffer.FloatBuffer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Arrays;

/**
 * Class tests tokenization and model prediction of BDS
 * These is not a full working solution, rather an experiment and PoC
 */
public class ModelLoader {
    public static void testTokenization() throws IOException, InvalidKerasConfigurationException {

        //Load json exported from python pickled code
        String path = "tokenizerLowEquippedAircraft.json";
        KerasTokenizer tokenizer = KerasTokenizer.fromJson(Resources.asFile(path).getAbsolutePath());

        //Sample messages to be tokenized
        String [] input = new String[] {
                "cf ba 33 20 60 17 fe",
                "95 80 00 30 a4 00 00",
                "80 39 d7 26 00 04 a3",
                "cf ba 27 1f ff ef fe"};

        System.out.println("Tokens: ");
        System.out.println(Arrays.deepToString(tokenizer.textsToSequences(input)));
        //Expecting these values (the same were returned by the python code)
        //[[190, 165, 144, 13, 24, 44, 10],
        // [114, 5, 1, 3, 25, 1, 1],
        // [5, 78, 138, 191, 1, 4, 113],
        // [190, 165, 124, 62, 9, 183, 10]]
    }


    /**
     * Method predicts BDS type basing on the single tokenized message
     */
    public static int[] calculateSingle(float[] msgTokens, MultiLayerNetwork model) {

        NDArray ndArray = new NDArray(1, 7);
        DataBuffer dataBuffer = new FloatBuffer(msgTokens);
        ndArray = new NDArray(1, 7);
        ndArray.setData(dataBuffer);
        return model.predict(ndArray);
    }

    /**
     * Method predicts BDS type basing on the batch of tokenized messages
     */
    public static int[] calculateBatch(INDArray a, MultiLayerNetwork model) {
        return model.predict(a);
    }

    public static void main(String args[]) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {

        //String simpleMlp = new ClassPathResource("LowEquipedAircraft.h5").getFile().getPath();
        //String json = Resources.asFile("model.json").getAbsolutePath();

        testTokenization();

        String h5 = Resources.asFile("simple_mlp.h5").getAbsolutePath();
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(h5);
        model.init();
        System.out.println(model.summary());

        //Build up the vector to calculate BDS
        //In real life we would first need to translate the byte array into tokens
        float[][] tokens = {{151, 92, 46, 15, 16, 110, 5},
                {128, 9, 1, 2, 8, 1, 1},
                {9, 38, 109, 125, 1, 6, 143},
                {151, 92, 96, 75, 4, 71, 5}};

        //Prepare vector in INDArray - the model expects this format
        INDArray initCondition = Nd4j.zeros(10000, 7);
        for (int i = 0; i < 10000; i++) {
            for (int j=0; j<7; j++) {
                initCondition.putScalar(i, j, tokens[i%4][j]);
            }
        }


        System.out.println("Calculating 10_000 batch messages");
        System.out.println("Starting at [millis]: " + System.currentTimeMillis());
        int[] ints = calculateBatch(initCondition, model);
        System.out.println("Finished at: [millis]: " + System.currentTimeMillis());

        //Expecting repeating output of 3 1 2 3 .... 3 1 2 3 ... (the same output as observed in python)
        System.out.println(Arrays.toString(ints));


        System.out.println("Calculating 100 single messages");
        System.out.println("Starting at [millis]: " + System.currentTimeMillis());
        int[] result = null;
        for (int i = 0; i < 100; i++) {
            result = calculateSingle(tokens[i%4], model);
        }
        System.out.println("Finished at: [millis]: " + System.currentTimeMillis());
        System.out.println("Last result " + Arrays.toString(result));

    }
}

