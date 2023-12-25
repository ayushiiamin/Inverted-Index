import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.StringTokenizer;
import java.util.StringJoiner;
import java.util.Set;

public class UnigramInvertedIdx {
    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {
        private Text id = new Text();
        private Text word = new Text();
      
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String sent = value.toString();
            String[] sentStuff = sent.split("\t", 2);

            String compSent = sentStuff[1];
            String lowerCaseSent = compSent.toLowerCase();
            lowerCaseSent = lowerCaseSent.replaceAll("[^a-z\\s]", " ");
          
            StringTokenizer tkn = new StringTokenizer(lowerCaseSent);

            String docID = sentStuff[0];
            id.set(docID);
          
            while (tkn.hasMoreTokens()) {
                word.set(tkn.nextToken());
                context.write(word, id);
            }
        }
    }

    public static class IndexReducer extends Reducer<Text, Text, Text, Text> {
      
        private Text finalRes = new Text();
      
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
          
            HashMap<String, Integer> mapOfFreq = new HashMap<String, Integer>();
          
            for (Text value : values) {
                String docID = value.toString();
                Integer frequenceyCurrUpdat = (mapOfFreq.getOrDefault(docID, 0)+ 1);
              
                mapOfFreq.put(docID, frequenceyCurrUpdat);
            }

            StringJoiner resStr = new StringJoiner("\t");
            Set<String> keys = mapOfFreq.keySet();
          
            for (String str : keys) {
                  resStr.add(str + ":" + mapOfFreq.get(str));
            }

            String finalTxt = resStr.toString();

            finalRes.set(finalTxt);
            context.write(key, finalRes);

        }
    }

    public static void main(String[] args) throws Exception {

        Job newJb = new Job();
        newJb.setJobName("Unigram Inverted Idx");
        newJb.setJarByClass(UnigramInvertedIdx.class);

        FileInputFormat.addInputPath(newJb, new Path(args[0]));
        FileOutputFormat.setOutputPath(newJb, new Path(args[1]));
        
        newJb.setMapperClass(TokenizerMapper.class);
        newJb.setReducerClass(IndexReducer.class);

        newJb.setOutputKeyClass(Text.class);
        newJb.setOutputValueClass(Text.class);

        if (newJb.waitForCompletion(true)) {
          System.exit(0);
        }
        else {
          System.exit(1);
        }
    }
}