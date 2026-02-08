import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.estimators.Estimator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

public class ExportWekaNaiveBayes {
  public static void main(String[] args) throws Exception {
    if (args.length != 3) {
      System.err.println("Usage: ExportWekaNaiveBayes <modelPath> <arffPath> <outJson>");
      System.exit(1);
    }

    String modelPath = args[0];
    String arffPath = args[1];
    String outJson = args[2];

    Classifier cls = (Classifier) SerializationHelper.read(modelPath);
    if (!(cls instanceof NaiveBayes)) {
      throw new IllegalArgumentException("Model is not NaiveBayes: " + cls.getClass().getName());
    }

    NaiveBayes nb = (NaiveBayes) cls;

    Instances data = DataSource.read(arffPath);
    if (data.classIndex() < 0) {
      data.setClassIndex(data.numAttributes() - 1);
    }

    Attribute classAttr = data.classAttribute();
    List<String> classVals = new ArrayList<>();
    for (int i = 0; i < classAttr.numValues(); i++) {
      classVals.add(classAttr.value(i));
    }

    Estimator classDist = (Estimator) getField(nb, "m_ClassDistribution");
    Estimator[][] dists = (Estimator[][]) getField(nb, "m_Distributions");

    StringBuilder json = new StringBuilder();
    json.append("{\n");
    json.append("  \"classAttribute\": ").append(q(classAttr.name())).append(",\n");
    json.append("  \"classes\": ").append(qList(classVals)).append(",\n");

    json.append("  \"classPriors\": {\n");
    for (int i = 0; i < classVals.size(); i++) {
      String clsName = classVals.get(i);
      double p = getProb(classDist, i);
      json.append("    ").append(q(clsName)).append(": ").append(d(p));
      if (i < classVals.size() - 1) json.append(",");
      json.append("\n");
    }
    json.append("  },\n");

    json.append("  \"attributes\": [\n");
    int outCount = 0;
    for (int a = 0; a < data.numAttributes(); a++) {
      if (a == data.classIndex()) continue;
      Attribute attr = data.attribute(a);
      List<String> values = new ArrayList<>();
      for (int v = 0; v < attr.numValues(); v++) {
        values.add(attr.value(v));
      }

      json.append("    {\n");
      json.append("      \"name\": ").append(q(attr.name())).append(",\n");
      json.append("      \"values\": ").append(qList(values)).append(",\n");
      json.append("      \"condProbs\": {\n");

      for (int c = 0; c < classVals.size(); c++) {
        String clsName = classVals.get(c);
        json.append("        ").append(q(clsName)).append(": [");
        for (int v = 0; v < values.size(); v++) {
          double p = getProb(dists[a][c], v);
          json.append(d(p));
          if (v < values.size() - 1) json.append(", ");
        }
        json.append("]");
        if (c < classVals.size() - 1) json.append(",");
        json.append("\n");
      }

      json.append("      }\n");
      json.append("    }");

      outCount++;
      if (outCount < data.numAttributes() - 1) json.append(",");
      json.append("\n");
    }

    json.append("  ],\n");
    json.append("  \"epsilon\": 1e-12\n");
    json.append("}\n");

    try (BufferedWriter bw = new BufferedWriter(new FileWriter(outJson))) {
      bw.write(json.toString());
    }

    System.out.println("Wrote: " + outJson);
  }

  private static Object getField(Object obj, String fieldName) throws Exception {
    Field f = obj.getClass().getDeclaredField(fieldName);
    f.setAccessible(true);
    return f.get(obj);
  }

  private static double getProb(Estimator est, int idx) throws Exception {
    Method m = null;
    for (Method cand : est.getClass().getMethods()) {
      if (!cand.getName().equals("getProbability")) continue;
      Class<?>[] p = cand.getParameterTypes();
      if (p.length == 1 && (p[0] == int.class || p[0] == double.class)) {
        m = cand;
        break;
      }
    }
    if (m == null) throw new IllegalStateException("Estimator has no getProbability method");
    Object val = (m.getParameterTypes()[0] == int.class) ? m.invoke(est, idx) : m.invoke(est, (double) idx);
    return ((Number) val).doubleValue();
  }

  private static String q(String s) {
    return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
  }

  private static String qList(List<String> items) {
    StringBuilder b = new StringBuilder("[");
    for (int i = 0; i < items.size(); i++) {
      b.append(q(items.get(i)));
      if (i < items.size() - 1) b.append(", ");
    }
    b.append("]");
    return b.toString();
  }

  private static String d(double v) {
    if (Double.isNaN(v) || Double.isInfinite(v)) return "0";
    return Double.toString(v);
  }
}
